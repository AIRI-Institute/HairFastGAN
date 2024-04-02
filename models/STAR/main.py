import argparse
from trainer import train
from tester import test


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--image_dir", type=str, default=None, help="the directory of image")
    group.add_argument("--annot_dir", type=str, default=None, help="the directory of annot")


def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument("--mode", type=str, default="train", help="train or test")
    group.add_argument("--config_name", type=str, default="alignment", help="set configure file name")
    group.add_argument('--device_ids', type=str, default="0,1,2,3",
                       help="set device ids, -1 means use cpu device, >= 0 means use gpu device")
    group.add_argument('--data_definition', type=str, default='WFLW', help="COFW, 300W, WFLW")
    group.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    group.add_argument("--batch_size", type=int, default=128, help="the batch size in train process")
    group.add_argument('--width', type=int, default=256, help='the width of input image')
    group.add_argument('--height', type=int, default=256, help='the height of input image')


def add_train_options(parser):
    group = parser.add_argument_group('train')
    group.add_argument("--train_num_workers", type=int, default=None, help="the num of workers in train process")
    group.add_argument('--loss_func', type=str, default='STARLoss_v2', help="loss function")
    group.add_argument("--val_batch_size", type=int, default=None, help="the batch size in val process")
    group.add_argument("--val_num_workers", type=int, default=None, help="the num of workers in val process")


def add_eval_options(parser):
    group = parser.add_argument_group("eval")
    group.add_argument("--pretrained_weight", type=str, default=None,
                       help="set pretrained model file name, if ignored then train the network without pretrain model")
    group.add_argument('--norm_type', type=str, default='default', help='default, ocular, pupil')
    group.add_argument('--test_file', type=str, default="test.tsv", help='for wflw, test.tsv/test_xx_metadata.tsv')


def add_starloss_options(parser):
    group = parser.add_argument_group('starloss')
    group.add_argument('--star_w', type=float, default=1, help="regular loss ratio")
    group.add_argument('--star_dist', type=str, default='smoothl1', help='STARLoss distance function')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry Function")
    add_base_options(parser)
    add_data_options(parser)
    add_train_options(parser)
    add_eval_options(parser)
    add_starloss_options(parser)

    args = parser.parse_args()
    print(args)

    print(
        "mode is %s, config_name is %s, pretrained_weight is %s, image_dir is %s, annot_dir is %s, device_ids is %s" % (
            args.mode, args.config_name, args.pretrained_weight, args.image_dir, args.annot_dir, args.device_ids))
    args.device_ids = list(map(int, args.device_ids.split(",")))
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        print("unknown running mode")
