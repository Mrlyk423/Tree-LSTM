"""
Preprocessing script for SemEval-2010 Task 8 data.

"""

import os
import glob

#
# Trees and tree loading
#

import os
import glob

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)

def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
        % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'triple.txt'), 'w') as triplefile, \
         open(os.path.join(dst_dir, 'sents.txt'), 'w') as sentsfile:
            for line in datafile:
                e1, e2, rel, sent = line.strip().split('\t')
                triplefile.write(e1 + "\t" + e2 + "\t" + rel + '\n')
                sentsfile.write(sent + '\n')

def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'sents.txt'), cp=cp, tokenize=True)
    constituency_parse(os.path.join(dirpath, 'sents.txt'), cp=cp, tokenize=True)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing SemEval-2010 Task 8 dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    rel_class_dir = os.path.join(data_dir, 'rel-class')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(rel_class_dir, 'train')
    test_dir = os.path.join(rel_class_dir, 'test')
    make_dirs([train_dir, test_dir])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    # split into separate files
    split(os.path.join(rel_class_dir, 'train.txt'), train_dir)
    split(os.path.join(rel_class_dir, 'test.txt'), test_dir)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(test_dir, cp=classpath)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(rel_class_dir, '*/*.toks')),
        os.path.join(rel_class_dir, 'vocab.txt'))
    build_vocab(
        glob.glob(os.path.join(rel_class_dir, '*/*.toks')),
        os.path.join(rel_class_dir, 'vocab-cased.txt'),
        lowercase=False)

