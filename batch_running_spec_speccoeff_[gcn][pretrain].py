from pre_train import process,GPTConfig
import numpy as np
import os
if __name__ == '__main__':
    os.makedirs('log_spec_speccoeff_[gcn]', exist_ok=True)

    #n layer
    os.makedirs('log_spec_speccoeff_[gcn][pretrain]/layer', exist_ok=True)
    final_metrics = {'AUC': [], 'sensitivity': [], 'specificity': [], 'accuracy': [], 'f1': []}
    for layer in [4,6,8]:
        for seed in range(7,10):
            config = dict(block_size=15, n_layer=layer, n_head=16, n_embd=32, dropout=0.15, bias=True,
                          cls_num=5,
                          apply_ehr=True,
                          apply_gene=True,
                          apply_mri=True,
                          apply_blood_test=True,
                          apply_protein=True,
                          apply_neurological_test=True,
                          apply_spec=True,
                          apply_spec_meta=True,
                          spec_norm=False,
                          if_data_balance = True,
                          spectra_encoder = 'gcn',
                          device = 'cuda:0',
                          add_noise = True,
                          )

            config = GPTConfig(**config)
            path = 'log_spec_speccoeff_[gcn][pretrain]/layer/' + str(layer) + '/' + str(seed)

            if os.path.exists(path):
                p = process(config, seg_seed=seed, lr=1e-3, batch_size=64, save_path=path)
                p.load_model()
                # p.pretrain(3)
                # p.train(50)
            else:
                p = process(config, seg_seed=seed, lr=1e-3, batch_size=64, save_path=path)
                p.pretrain(100)
                p.train(100)
            final_metric = p.save_everything()
            for k in final_metric:
                final_metrics[k].append([layer, final_metric[k]])

            np.save('log_spec_speccoeff_[gcn][pretrain]/layer/final_metrics.npy', final_metrics)

    print(final_metrics)

    # n embed
    os.makedirs('log_spec_speccoeff_[gcn][pretrain]/embd', exist_ok=True)
    final_metrics = {'AUC': [], 'sensitivity': [], 'specificity': [], 'accuracy': [], 'f1': []}
    for embd in [64, 128, 256]:
        for seed in range(8):
            config = dict(block_size=15, n_layer=8, n_head=16, n_embd=embd, dropout=0.2, bias=True,
                          cls_num=3,
                          apply_ehr=True,
                          apply_gene=True,
                          apply_mri=True,
                          apply_blood_test=True,
                          apply_protein=True,
                          apply_neurological_test=True,
                          apply_spec=True,
                          apply_spec_meta=True,
                          spec_norm=False,
                          if_data_balance = True,
                          spectra_encoder = 'gcn')
            config = GPTConfig(**config)
            path = 'log_spec_speccoeff_[gcn][pretrain]/embd/' + str(embd) + '/' + str(seed)

            if os.path.exists(path):
                p = process(config, seg_seed=seed, lr=1e-3, batch_size=64, save_path=path)
                p.load_model()
                p.pretrain(3)
                p.train(50)
            else:
                p = process(config, seg_seed=seed, lr=1e-3, batch_size=64, save_path=path)
                p.pretrain(100)
                p.train(50)
            final_metric = p.save_everything()
            for k in final_metric:
                final_metrics[k].append([embd, final_metric[k]])

            np.save('log_spec_speccoeff_[gcn][pretrain]/embd/final_metrics.npy', final_metrics)

    print(final_metrics)


