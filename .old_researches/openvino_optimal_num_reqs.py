#!/usr/bin/env python3

import argparse
from openvino.inference_engine import IENetwork, IECore

def parse_args(args):
    parser = argparse.ArgumentParser(description='network files')
    parser.add_argument(
        '--bin',
        help='path to bin openVINO inference model',
        type=str,
        required=True
    )
    parser.add_argument(
        '--xml',
        help='path to xml model sheme',
        type=str,
        required=True
    )

    return parser.parse_args(args)

def main(args=None):
    args = parse_args(args)
    model_xml = args.xml
    model_bin = args.bin

    core = IECore()
    net = core.read_network(model=model_xml, weights=model_bin)

    devices = ['CPU', 'GPU', 'HETERO:GPU,CPU', 'HETERO:CPU,GPU', 'MULTI:CPU,GPU', 'MULTI:GPU,CPU']

    for device in devices:
        exec_net = core.load_network(network=net, device_name=device)
        num_req = exec_net.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        print("Device:", device, 'num_req:', num_req)


if __name__ == '__main__':
    main()