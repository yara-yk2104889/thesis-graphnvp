from time import time
from utils.argparser import args
import os, sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join as joinpath
from rdkit import Chem
from torch.utils.data import DataLoader, Subset
from graph_nvp.hyperparams import Hyperparameters
from graph_nvp.nvp_model import GraphNvpModel
from utils import environment as env
from utils.data_utils import construct_mol, save_mol_png, NumpyTupleDataset
from tqdm import tqdm

def read_molecules(path):
    f = open(joinpath(path, '{}_config.txt'.format(args.data_name)), 'r')
    data_config = eval(f.read())
    f.close()
    fp = open(joinpath(args.data_dir, '{}_kekulized_ggnp.txt'.format(args.data_name)), 'r')
    all_smiles = [smiles.strip() for smiles in fp]
    fp.close()
    return data_config, all_smiles

def sample_z(model, batch_size=args.batch_size, z_mu=None):
    z_dim = model.adj_size + model.x_size
    mu = np.zeros([z_dim], dtype=np.float32)
    sigma_diag = np.ones([z_dim])
    sigma_diag = np.sqrt(np.exp(model.prior_ln_var.item())) * sigma_diag
    sigma = args.temp * sigma_diag

    if z_mu is not None:
        mu = z_mu
        sigma = 0.01 * np.eye(z_dim, dtype=np.float32)

    z = np.random.normal(mu, sigma, (batch_size, z_dim)).astype(np.float32)
    z = torch.from_numpy(z).float().to(args.device)
    return z.detach()

def main():
    args.cuda = torch.cuda.is_available()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args)
    if args.data_name == 'qm9':
        from data import transform_qm9
        transform_fn = transform_qm9.transform_fn
        args.atomic_num_list = [6, 7, 8, 9, 0]
        mlp_channels = [256, 256]
        gnn_channels = {'gcn': [8, 64], 'hidden': [128, 64]}  # official QM9 sizes
        valid_idx = transform_qm9.get_val_ids()
    elif args.data_name == 'zinc250k':
        from data import transform_zinc250k
        from data.transform_zinc250k import transform_fn_zinc250k
        transform_fn = transform_fn_zinc250k
        args.atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        mlp_channels = [1024, 512]
        gnn_channels = {'gcn': [16, 128], 'hidden': [256, 64]}  # official ZINC sizes
        valid_idx = transform_zinc250k.get_val_ids()

    dataset = NumpyTupleDataset(joinpath(args.data_dir, args.data_file), transform_fn)

    # Same split as chainer's split_dataset / split_dataset_random
    if len(valid_idx) > 0:
        valid_set = set(valid_idx)
        train_idx = [t for t in range(len(dataset)) if t not in valid_set]
        train, test = Subset(dataset, train_idx), Subset(dataset, valid_idx)
    else:
        order = np.random.RandomState(args.seed).permutation(len(dataset))
        n_train = int(len(dataset) * 0.8)
        train, test = Subset(dataset, order[:n_train]), Subset(dataset, order[n_train:])
    if args.subset > 0:
        train = Subset(train, range(min(args.subset, len(train))))
        test = Subset(test, range(min(max(1, args.subset // 10), len(test))))
        print('Using a subset: {} training, {} validation molecules'.format(len(train), len(test)))

    num_masks = {'node': args.num_node_masks, 'channel': args.num_channel_masks}
    mask_size = {'node': args.node_mask_size, 'channel': args.channel_mask_size}
    num_coupling = {'node': args.num_node_coupling, 'channel': args.num_channel_coupling}
    NVPmodel_params = Hyperparameters(args.num_atoms, args.num_rels, len(args.atomic_num_list),
                                   num_masks=num_masks, mask_size=mask_size, num_coupling=num_coupling,
                                   batch_norm=args.apply_batch_norm,
                                   additive_transformations=True,  # the model always uses additive couplings
                                   mlp_channels=mlp_channels,
                                   gnn_channels=gnn_channels)
    model = GraphNvpModel(NVPmodel_params).to(args.device)
    # shuffle=True draws a new order every epoch (seeded by torch.manual_seed)
    dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    data_config, all_train_smiles = read_molecules(args.data_dir)

    # fitting
    t_total = time()
    total_g_loss, total_d_loss = [], []
    max_size = model.num_atoms  # 9 for QM9
    num_atom = max_size
    node_dim = model.num_features  # 5 for QM9 # OR exclude padding dim. 5-1
    bond_dim = model.num_bonds  # 4 for QM9
    best_g_loss, best_d_loss = sys.maxsize, sys.maxsize
    start_epoch = args.resume_epoch
    resume_optimizer_state = None
    if args.resume:
        model = GraphNvpModel(hyperparams=NVPmodel_params).to(args.device)
        model_path = joinpath(args.model_save_dir, 'epoch{}-mle.ckpt'.format(args.resume_epoch))
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model' in checkpoint and 'optimizer' in checkpoint:
            # Full checkpoint: model weights, Adam's state and the epoch number
            model.load_state_dict(checkpoint['model'])
            resume_optimizer_state = checkpoint['optimizer']
            start_epoch = checkpoint['epoch']
        else:
            # Older checkpoint that holds only the model weights
            model.load_state_dict(checkpoint)
            print("Warning: {} has no optimizer state; Adam's state starts fresh".format(model_path))
        print("Resuming from epoch{}".format(start_epoch))

    all_unique_rate = []
    all_valid_rate = []
    all_novelty_rate = []
    print('start fitting.')

    # Plain Adam with a constant learning rate, as in the official code (no LR schedule)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.mle_lr, betas=(args.beta1, args.beta2))
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    optimizer.step()

    def generate_one(model, mute=False, cnt=None):
        """
        inverse flow to generate one molecule
        Args:
            temp: temperature of normal distributions, we sample from (0, temp^2 * I)
        """
        generate_start_t = time()
        num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
        num2bond_symbol = {0: '=', 1: '==', 2: '==='}
        num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
        num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}
        is_continue = True
        mol = None
        total_resample = 0
        batch_size = 1
        # Generating
        z = sample_z(model, batch_size=1)
        A, X = model.reverse(z, model.x_size) # For QM9: [16,9,9,5], [16,9,5], [16,8]-[B,z_dim]
        X = F.softmax(X, dim=2)
        mols = [construct_mol(x_elem, adj_elem, args.atomic_num_list)
                for x_elem, adj_elem in zip(X, A)]
        pure_valid = 0
        smiles = ''
        num_atoms = -1
        for mol in mols:
            assert mol is not None, 'mol is None...'
            final_valid = env.check_chemical_validity(mol)
            valency_valid = env.check_valency(mol)

            if final_valid is False or valency_valid is False:
                print('Warning: use valency check during generation but the final molecule is invalid!!!')
                continue
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            smiles = Chem.MolToSmiles(mol)

            if total_resample == 0:
                pure_valid = 1.0
            if not mute:
                cnt = str(cnt) if cnt is not None else ''
                print('smiles%s: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (
                    cnt, smiles, num_atoms, num_bonds, total_resample, time() - generate_start_t))
        return smiles, A, X, pure_valid, num_atoms

    def validation_nll(model):
        """
        Same loss as in training, on the clean validation molecules: eval mode (no A noise,
        BatchNorm running statistics), no X noise, no gradients
        """
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for X, A, label in valid_loader:
                X, A = X.float().to(args.device), A.float().to(args.device)
                z, sum_log_det_jacs = model(A, X)
                # log_prob averages over the batch; weight by batch size for the mean over all molecules
                total += model.log_prob(z, sum_log_det_jacs).item() * X.shape[0]
                count += X.shape[0]
        return total / count

    loss_csv = joinpath(args.model_save_dir, 'losses.csv')

    def start_loss_csv():
        # Keep rows up to the epoch we resume from, so a resumed run does not duplicate epochs
        rows = []
        if start_epoch > 0 and os.path.exists(loss_csv):
            with open(loss_csv) as f:
                rows = [line for line in f.read().splitlines()[1:] if int(line.split(',')[0]) <= start_epoch]
        os.makedirs(args.model_save_dir, exist_ok=True)
        with open(loss_csv, 'w') as f:
            f.write('epoch,train_loss,valid_nll\n')
            for line in rows:
                f.write(line + '\n')

    def train(model):
        start_loss_csv()
        # Continue from the epoch after the checkpoint up to --epochs (inclusive)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            # gen() at the end of each epoch calls model.eval(); switch back so BatchNorm trains
            model.train()
            batch_g_losses = []
            batch_cnt = 0
            epoch_example = 0
            num_batches = len(dataloader)
            pbar = tqdm(total=num_batches)

            for i_batch, batch_data in enumerate(dataloader):
                batch_time_s = time()
                loss = {}
                X, A, label = batch_data[0], batch_data[1], batch_data[2]
                X, A, label = X.float().to(args.device), \
                              A.float().to(args.device), \
                              label.float().to(args.device)
                # Dequantization of X; A is dequantized inside the model, after the X-couplings
                X_prime = X + 0.9 * torch.rand(X.shape, device=args.device)
                z, sum_log_det_jacs = model(A, X_prime)
                nll = model.log_prob(z, sum_log_det_jacs)
                g_loss = nll
                loss['G/loss_g'] = g_loss.item()
                batch_g_losses.append(g_loss.item())
                optimizer.zero_grad()
                g_loss.backward()
                optimizer.step()
                pbar.update()
                if i_batch % args.show_loss_step == 0:
                    tqdm.write("Epoch %d, batch %d, Loss mle: %.5f" % (epoch, i_batch, g_loss.item()))
            pbar.close()

            train_loss = float(np.mean(batch_g_losses))
            valid_nll = validation_nll(model)
            print('Epoch %d | train loss %.5f | validation NLL %.5f' % (epoch, train_loss, valid_nll))
            with open(loss_csv, 'a') as f:
                f.write('%d,%.6f,%.6f\n' % (epoch, train_loss, valid_nll))

            if epoch % args.save_epochs == 0 or epoch == args.epochs:
                print("Saving GraphNVP model trained with maximum liklihood")
                os.makedirs(args.model_save_dir, exist_ok=True)
                model_path = joinpath(args.model_save_dir, 'epoch{}-mle.ckpt'.format(epoch))
                # hyperparams and train_args let evaluate.py rebuild exactly this model
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                            'hyperparams': vars(NVPmodel_params), 'train_args': vars(args)},
                           model_path)
                print('Saved model checkpoints into {}...'.format(args.model_save_dir))
            gen(model, epoch)

    def gen(model, epoch=-1):
        model.eval()
        all_smiles = []
        pure_valids = []
        appear_in_train = 0.
        start_t = time()
        cnt_mol = 0
        cnt_gen = 0
        out_path = joinpath(args.gen_path, 'mle_mols{}.txt'.format(epoch))
        print("Generating %d mols for evaluation" % (args.num_gen))
        while cnt_mol < args.num_gen:
            smiles, A, X, no_resample, num_atoms = generate_one(model, mute=False, cnt=cnt_gen)
            cnt_gen += 1
            if cnt_gen > args.max_resample:
                break
            if num_atoms < 0 or num_atoms < args.min_atoms:
                print('#atoms of generated molecule less than %d, discarded!' % args.min_atoms)
                continue
            else:
                cnt_mol += 1
                if cnt_mol % 100 == 0:
                    print('cur cnt mol: %d' % cnt_mol)
                all_smiles.append(smiles)
                pure_valids.append(no_resample)
                print('Accepting: {}'.format(smiles))
                if all_train_smiles is not None and smiles in all_train_smiles:
                    appear_in_train += 1.0
            mol = Chem.MolFromSmiles(smiles)
            qed_score = env.qed(mol)
            plogp_score = env.penalized_logp(mol)
        if cnt_mol > args.num_gen:
            print("Generating {} times rather than 100 times!".format(cnt_mol))
            args.num_gen = cnt_mol

        unique_smiles = list(set(all_smiles))
        unique_rate = len(unique_smiles) / args.num_gen
        pure_valid_rate = sum(pure_valids) / args.num_gen
        novelty = 1. - (appear_in_train / args.num_gen)

        print('Time for generating (%d/%d) molecules(#atoms>=%d) with %d resamplings: %.5f' % (
               cnt_gen-args.max_resample, args.num_gen, args.min_atoms, args.max_resample, time() - start_t))
        print('| unique rate: %.5f | valid rate: %.5f | novelty: %.5f |' % (unique_rate, pure_valid_rate, novelty))
        mol_img_dir = joinpath(args.img_dir, 'mol_img{}'.format(epoch))
        os.makedirs(mol_img_dir, exist_ok=True)
        if not os.path.exists(args.gen_path):
            os.makedirs(args.gen_path)
        if out_path is not None and args.save:
            with open(out_path, 'w+') as out_file:
                cnt = 0
                for i, mol in enumerate(all_smiles):
                    # Invalid disconnection
                    if '.' in all_smiles[i]:
                        continue
                    out_file.write(all_smiles[i] + '\n')
                    save_mol_png(Chem.MolFromSmiles(mol), joinpath(mol_img_dir, '{}.png'.format(i)))
                    cnt += 1
            print('writing %d smiles into %s done!' % (cnt, out_path))
        all_unique_rate.append(unique_rate)
        all_valid_rate.append(pure_valid_rate)
        all_novelty_rate.append(novelty)
        if args.save: 
            print('saving metric of validity, novelty and uniqueness into %s' %(args.gen_path))
            np.save(joinpath(args.gen_path, 'valid{}'.format(epoch)), np.array(all_valid_rate))
            np.save(joinpath(args.gen_path, 'novelty{}'.format(epoch)), np.array(all_novelty_rate))
            np.save(joinpath(args.gen_path, 'unique{}'.format(epoch)), np.array(all_unique_rate))

    if args.mode == 'train':
        train(model)
    elif args.mode == 'gen':
        gen(model)
    else:
        print("Specify mode as 'train' or 'gen'")

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main()