#!/usr/bin/env python

import sys
import argparse
import logging

from transformers import AutoTokenizer

# local libs
from models import load_quantized_gptq, load_quantized
from vector_stores import VectorStores
from utils import documents_loader, make_chain
from webui import WebApp

LOG = logging.getLogger(__name__)

def set_logging(args):
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(log_level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    formatter = logging.Formatter('%(levelname)-8s- %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)


def arg_parse():
    parser = argparse.ArgumentParser(
        description='main script'
    )

    # global parser
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument('--model-dir',
                        type=str,
                        help='model dir',
                        default='/home/siegfried/text-generation-webui/models')

    parser.add_argument('--redis-url',
                        type=str,
                        help='redis url',
                        default='redis://localhost:6379')

    parser.add_argument('--index-name',
                        type=str,
                        help='index name',
                        required=True)

    parser.add_argument('--encode-model',
                        type=str,
                        help='encode model',
                        default='sentence-transformers_all-MiniLM-L6-v2')

    parser.add_argument('--debug',
                        action='store_true',
                        help='debug')

    # store parser
    store_parser = subparsers.add_parser("store")

    store_parser.add_argument('--docs',
                              type=str,
                              nargs='+',
                              help='a list of files, only takes txt and pdf files')

    store_parser.add_argument('--chunk-size',
                              type=int,
                              help='chunk size',
                              default=500)

    store_parser.add_argument('--chunk-overlap',
                              type=int,
                              help='chunk overlap',
                              default=100)

    # run parser
    run_parser = subparsers.add_parser("run")

    run_parser.add_argument('--model-name',
                            type=str,
                            help='model name',
                            required=True)

    run_parser.add_argument('--model-type',
                            type=str,
                            help='model type',
                            default='llama')

    run_parser.add_argument('--wbits',
                            type=int,
                            help='wbits',
                            default=4)

    run_parser.add_argument('--groupsize',
                            type=int,
                            help='groupsize',
                            default=128)

    run_parser.add_argument('--no-gptq',
                            action='store_true',
                            help='if model is NOT gptq quantized.')

    run_parser.add_argument('--pre-layer',
                            type=int,
                            help='pre layer')

    run_parser.add_argument('--gpu-memory',
                            type=int,
                            help='gpu memory')

    run_parser.add_argument('--cpu-memory',
                            type=int,
                            help='cpu memory')

    run_parser.add_argument('--cpu',
                            action='store_true',
                            help='cpu')

    run_parser.add_argument('--share',
                            action='store_true',
                            help='share or not')

    args = parser.parse_args()
    
    return args

def main():
    args = arg_parse()

    set_logging(args)

    vs = VectorStores(redis_url=args.redis_url,
                      index_name=args.index_name,
                      model_dir=args.model_dir,
                      encode_model=args.encode_model)

    if args.command == "store":
        splited_docs = documents_loader(args.docs,
                                        chunk_size=args.chunk_size,
                                        chunk_overlap=args.chunk_overlap)
        vs.store(splited_docs)
        LOG.info("finished, exiting...")
        sys.exit(0)

    elif args.command == "run":
        rds = vs.load()

        model_path = f"{args.model_dir}/{args.model_name}"

        LOG.info(f"Loading Tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")

        LOG.info(f"Loading the model from {model_path}...")
        if args.no_gptq:
            model = load_quantized(args.model_name, args)
        else:
            LOG.info("Loading GPTQ models...")
            model = load_quantized_gptq(args.model_name, args)

        chain = make_chain(model, tokenizer, args)

        app = WebApp(rds=rds,
                     chain=chain,
                     args=args)
        app.run()

if __name__ == "__main__":
    main()
