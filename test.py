import argparse
import evaluate_MemNet
import warnings
warnings.filterwarnings("ignore")

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)
    parser.add_argument("--preds", type=int, default=5)

    parser.add_argument("--model", default=r'training/training_controller_gcn/2022-09-13 09_/model_controller_gcn_epoch_139_2022-09-13 09')
    parser.add_argument("--saved_memory", default=True)
    parser.add_argument("--memories_path", default=r'test/2023-02-14 11_run time and space test/')
    parser.add_argument("--online_learning", default=True)

    parser.add_argument("--dataset_file", default="kitti_dataset.json", help="dataset file")
    parser.add_argument("--train_path", default=r'interaction/graph_dataset/DR_USA_Intersection_EP1_val.json')
    parser.add_argument("--val_path", default=r'interaction/graph_dataset/DR_USA_Intersection_EP1_val.json')
    parser.add_argument("--osm_path", default=r'D:\data\INTERACTION-Dataset-DR-multi-v1_2\maps\DR_USA_Intersection_EP1.osm')
    parser.add_argument("--withIRM", default=False, help='generate predictions with/without IRM')
    parser.add_argument("--saveImages", default=None,
                        help=
                        '''
                        Save in test folder examples of dataset with predictions generated by MANTRA.
                        If None, it does not save anything.
                        If 'All', it saves all examples.
                        If 'Subset', it saves examples defined in index_qualitative.py (handpicked significant samples)
                        ''')
    parser.add_argument("--visualize_dataset", default=False)
    parser.add_argument("--info", type=str, default='run time and space test'
                                                    '', help='Name of evaluation. '
                        'It will use for name of the test folder.')
    return parser.parse_args()


def main(config):
    v = evaluate_MemNet.Validator(config)
    print('start evaluation')
    v.test_model()


if __name__ == "__main__":
    config = parse_config()
    main(config)
