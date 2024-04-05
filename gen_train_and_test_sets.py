import argparse
from dataset_utils import generate_train_mixture, generate_competition_eval_mixture


def main(args):
    # Generate train mixture
    generate_train_mixture(args.soi_type, args.num_batches, args.batch_size)

    # Generate competition evaluation mixture
    generate_competition_eval_mixture(args.soi_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate mixtures for training and evaluation')
    parser.add_argument('--soi_type', type=str, choices=['QPSK', 'QPSK_OFDM'], help='Type of signal of interest (SOI)')
    parser.add_argument('--num_batches', type=int, help='Number of batches to generate for training')
    parser.add_argument('--batch_size', type=int, help='Batch size for generating mixtures')
    args = parser.parse_args()
    main(args)
