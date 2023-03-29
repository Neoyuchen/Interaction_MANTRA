import argparse
from trainer import trainer_ae_gcn


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=10, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=30, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    parser.add_argument("--dataset_file",
                        default=r"D:\data\INTERACTION-Dataset-DR-multi-v1_2\train\DR_USA_Roundabout_FT_train.csv",
                        help="dataset file")  # default="kitti_dataset.json"
    parser.add_argument("--val_file",
                        default=r"D:\data\INTERACTION-Dataset-DR-multi-v1_2\val\DR_USA_Roundabout_FT_val.csv",
                        help="dataset file")


    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    parser.add_argument("--pretrained_model", default=r"D:\Python_project\MANTRA-CVPR20\interaction\training\training_ae_gcn\2022-11-13 23_\model_ae_epoch_9_2022-11-13 23", help="path of model")
    parser.add_argument("--model_ae", default=r"training/training_ae_itr/2022-11-10 16_/model_ae_epoch_299_2022-11-10 16", help="path of autoencoder model")
    parser.add_argument("--neighbor_distance", default=5, help="max distance of interaction")

    return parser.parse_args()

def main(config):
    t = trainer_ae_gcn.Trainer(config)
    print('start training autoencoder with gcn layers')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)