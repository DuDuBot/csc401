import sys
import argparse
import os
import json
import re
import spacy
import html
import functools
import time

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment, steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment

    if 1 in steps:  # replace newlines with spaces
        # modComm = re.sub(r"\n{1,}", " ", modComm)
        modComm = modComm.replace('\n', '').replace('\r', '').replace('\t', '')
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces and punctuation
        modComm = re.sub(r'( {2,})', '', modComm)
        modComm = re.sub(r"(\s*$)", "", modComm)
    if modComm == '':  # every string should be terminated with a \n as per handout, even if empty.
        return '\n'
    utterance = nlp(modComm)
    modComm = ''
    for sent in utterance.sents:  # iterate over sentences
        if 6 in steps:  # POS tagging, stop word removal
            def new_str(pos):  # this is a generator that constructs a word+token using spacy's NLP
                for token in pos:
                    if token.lemma_[0] == '-' and token.text[0] != '-':  # as per handout
                        beg = token.text
                    else:
                        beg = token.lemma_
                    yield beg + '/' + token.tag_
            modComm += " ".join(new_str(sent)) + '\n' # end each sentence with '\n'
    return modComm


def main(args):
    allOutput = []
    indir = args.a1_dir
    stime = time.time()
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            if file not in ['Center', 'Right', 'Left', 'Alt']:
                continue
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            sindex = args.ID[0] % len(data)

            for i in range(sindex, sindex + args.max):
            # for i in range(0, 5):
                if i == len(data):  # Ensure circular indexing
                    i -= data
                if ((i+1) % 1000) == 0:
                    print(f"step: '{i+1}' at time '{time.time()-stime}'")
                line = json.loads(data[i])

                line_new = {key: line[key] for key in ['id', 'body']}
                line_new['cat'] = file
                line_new['body'] = preproc1(line_new['body'], steps=range(11))
                allOutput.append(line_new)
    output = args.output
    # output = os.path.join(indir, args.output)
    with open(output, 'w') as outfile:
      json.dump(allOutput, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'sample_outputs')
    main(args)
