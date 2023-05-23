import os
import json
import argparse
from tqdm import tqdm


def convert_collection(args):
    print('Converting collection...')
    with open(args.collection_path, 'r') as f:
        with open(os.path.join(args.output_folder, 'DPR_wikipedia_corpus.jsonl'), 'w') as fw:
            next(f)
            for line in tqdm(f):
                pid, text, title = line.split('\t')
                contents = title + text[1:-1]
                doc = json.dumps({"id": pid, "contents": contents})
                fw.write(doc + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NQ wikipedia tsv passage collection into jsonl files for Anserini.')
    parser.add_argument('--collection-path', required=True, help='Path to NQ wikipedia tsv collection.')
    parser.add_argument('--output-folder', required=True, help='Output folder.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    convert_collection(args)
    print('Done!')
