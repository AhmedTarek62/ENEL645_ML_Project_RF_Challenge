from dataset_utils import generate_train_mixture, generate_competition_eval_mixture

# QPSK Params
soi_type = 'QPSK'
num_batches = 50
batch_size = 128
generate_train_mixture(soi_type, num_batches, batch_size)
generate_competition_eval_mixture(soi_type)

# QPSK OFDM Params
soi_type = 'QPSK_OFDM'
num_batches = 50
batch_size = 128
generate_train_mixture(soi_type, num_batches, batch_size)
generate_competition_eval_mixture(soi_type)
