import argparse
from trainer import trainer_ae


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=300)

    parser.add_argument("--past_len", type=int, default=10, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=30, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--dataset_file", default= r"D:\data\INTERACTION-Dataset-DR-multi-v1_2\train\DR_USA_Roundabout_FT_train.csv", help="dataset file")  # default="kitti_dataset.json"
    parser.add_argument("--val_file", default= r"D:\data\INTERACTION-Dataset-DR-multi-v1_2\val\DR_USA_Roundabout_FT_val.csv", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    parser.add_argument("--pretrained_model", default=r"", help="path of model")

    return parser.parse_args()


def main(config):
    t = trainer_ae.Trainer(config)
    print('start training autoencoder')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
