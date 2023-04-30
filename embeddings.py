#!/usr/bin/env python

import sys
import argparse
import logging

from transformers import AutoTokenizer

from vector_stores import VectorStores
from models import load_normal, load_quantized
from utils import documents_loader, make_pipeline, make_chain
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

    parser.add_argument('--redis-url',
                        type=str,
                        help='redis url',
                        default='redis://localhost:6379')

    parser.add_argument('--index-name',
                        type=str,
                        help='index name',
                        required=True)

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


    run_parser.add_argument('--model-dir',
                        type=str,
                        help='model dir',
                        default='/home/siegfried/model-gptq')

    run_parser.add_argument('--model-name',
                            type=str,
                            help='model name',
                            required=True)

    run_parser.add_argument('--no-gptq',
                            action='store_true',
                            help='if model is NOT gptq quantized.')

    run_parser.add_argument('--use-safetensors',
                            action='store_true',
                            help='use fast load for tokenizer')

    run_parser.add_argument('--share',
                            action='store_true',
                            help='share or not')

    args = parser.parse_args()
    
    return args

def main():
    args = arg_parse()

    set_logging(args)

    vs = VectorStores(redis_url=args.redis_url,
                      index_name=args.index_name)

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
            model = load_normal(args.model_name, args)
        else:
            LOG.info("Loading gptq quantized models...")
            model = load_quantized(args.model_name, args)

        pipeline = make_pipeline(model, tokenizer)
        chain = make_chain(pipeline)

        app = WebApp(rds=rds,
                     chain=chain,
                     args=args)
        app.run()

if __name__ == "__main__":
    main()
