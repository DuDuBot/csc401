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
# common_abbreviations = [r"(?<!\bAla)", r'(?<!\bala)', r'(?<!\bALA)', r'(?<!\bAla)', r'(?<!\bAriz)', r'(?<!\bariz)',
#                                 r'(?<!\bARIZ)', r'(?<!\bAriz)', r'(?<!\bAssn)', r'(?<!\bassn)', r'(?<!\bASSN)',
#                                 r'(?<!\bAssn)', r'(?<!\bAtty)', r'(?<!\batty)', r'(?<!\bATTY)', r'(?<!\bAtty)',
#                                 r'(?<!\bAug)', r'(?<!\baug)', r'(?<!\bAUG)', r'(?<!\bAug)', r'(?<!\bAve)',
#                                 r'(?<!\bave)', r'(?<!\bAVE)', r'(?<!\bAve)', r'(?<!\bBldg)', r'(?<!\bbldg)',
#                                 r'(?<!\bBLDG)', r'(?<!\bBldg)', r'(?<!\bBlvd)', r'(?<!\bblvd)', r'(?<!\bBLVD)',
#                                 r'(?<!\bBlvd)', r'(?<!\bCalif)', r'(?<!\bcalif)', r'(?<!\bCALIF)', r'(?<!\bCalif)',
#                                 r'(?<!\bCapt)', r'(?<!\bcapt)', r'(?<!\bCAPT)', r'(?<!\bCapt)', r'(?<!\bCf)', r'(?<!\bcf)',
#                                 r'(?<!\bCF)', r'(?<!\bCf)', r'(?<!\bCh)', r'(?<!\bch)', r'(?<!\bCH)', r'(?<!\bCh)', r'(?<!\bCo)',
#                                 r'(?<!\bco)', r'(?<!\bCO)', r'(?<!\bCo)', r'(?<!\bCol)', r'(?<!\bcol)', r'(?<!\bCOL)',
#                                 r'(?<!\bCol)', r'(?<!\bColo)', r'(?<!\bcolo)', r'(?<!\bCOLO)', r'(?<!\bColo)', r'(?<!\bConn)',
#                                 r'(?<!\bconn)', r'(?<!\bCONN)', r'(?<!\bConn)', r'(?<!\bCorp)', r'(?<!\bcorp)', r'(?<!\bCORP)',
#                                 r'(?<!\bCorp)', r'(?<!\bDR)', r'(?<!\bdr)', r'(?<!\bDR)', r'(?<!\bDr)', r'(?<!\bDec)',
#                                 r'(?<!\bdec)', r'(?<!\bDEC)', r'(?<!\bDec)', r'(?<!\bDept)', r'(?<!\bdept)', r'(?<!\bDEPT)',
#                                 r'(?<!\bDept)', r'(?<!\bDist)', r'(?<!\bdist)', r'(?<!\bDIST)', r'(?<!\bDist)', r'(?<!\bDr)',
#                                 r'(?<!\bdr)', r'(?<!\bDR)', r'(?<!\bDr)', r'(?<!\bDrs)', r'(?<!\bdrs)', r'(?<!\bDRS)',
#                                 r'(?<!\bDrs)', r'(?<!\bEd)', r'(?<!\bed)', r'(?<!\bED)', r'(?<!\bEd)', r'(?<!\bEq)', r'(?<!\beq)',
#                                 r'(?<!\bEQ)', r'(?<!\bEq)', r'(?<!\bFEB)', r'(?<!\bfeb)', r'(?<!\bFEB)', r'(?<!\bFeb)', r'(?<!\bFeb)', r'(?<!\bfeb)',
#                                 r'(?<!\bFEB)', r'(?<!\bFeb)', r'(?<!\bFig)', r'(?<!\bfig)', r'(?<!\bFIG)', r'(?<!\bFig)', r'(?<!\bFigs)',
#                                 r'(?<!\bfigs)', r'(?<!\bFIGS)', r'(?<!\bFigs)', r'(?<!\bFla)', r'(?<!\bfla)', r'(?<!\bFLA)', r'(?<!\bFla)',
#                                 r'(?<!\bGa)', r'(?<!\bga)', r'(?<!\bGA)', r'(?<!\bGa)', r'(?<!\bGen)', r'(?<!\bgen)', r'(?<!\bGEN)', r'(?<!\bGen)',
#                                 r'(?<!\bGov)', r'(?<!\bgov)', r'(?<!\bGOV)', r'(?<!\bGov)', r'(?<!\bHON)', r'(?<!\bhon)', r'(?<!\bHON)', r'(?<!\bHon)',
#                                 r'(?<!\bIll)', r'(?<!\bill)', r'(?<!\bILL)', r'(?<!\bIll)', r'(?<!\bInc)', r'(?<!\binc)', r'(?<!\bINC)', r'(?<!\bInc)',
#                                 r'(?<!\bJR)', r'(?<!\bjr)', r'(?<!\bJR)', r'(?<!\bJr)', r'(?<!\bJan)', r'(?<!\bjan)', r'(?<!\bJAN)', r'(?<!\bJan)',
#                                 r'(?<!\bJr)', r'(?<!\bjr)', r'(?<!\bJR)', r'(?<!\bJr)', r'(?<!\bKan)', r'(?<!\bkan)', r'(?<!\bKAN)',
#                                 r'(?<!\bKan)', r'(?<!\bKy)', r'(?<!\bky)', r'(?<!\bKY)', r'(?<!\bKy)', r'(?<!\bLa)', r'(?<!\bla)', r'(?<!\bLA)',
#                                 r'(?<!\bLa)', r'(?<!\bLt)', r'(?<!\blt)', r'(?<!\bLT)', r'(?<!\bLt)', r'(?<!\bLtd)', r'(?<!\bltd)', r'(?<!\bLTD)',
#                                 r'(?<!\bLtd)', r'(?<!\bMR)', r'(?<!\bmr)', r'(?<!\bMR)', r'(?<!\bMr)', r'(?<!\bMRS)', r'(?<!\bmrs)', r'(?<!\bMRS)',
#                                 r'(?<!\bMrs)', r'(?<!\bMar)', r'(?<!\bmar)', r'(?<!\bMAR)', r'(?<!\bMar)', r'(?<!\bMass)', r'(?<!\bmass)',
#                                 r'(?<!\bMASS)', r'(?<!\bMass)', r'(?<!\bMd)', r'(?<!\bmd)', r'(?<!\bMD)', r'(?<!\bMd)', r'(?<!\bMessrs)',
#                                 r'(?<!\bmessrs)', r'(?<!\bMESSRS)', r'(?<!\bMessrs)', r'(?<!\bMich)', r'(?<!\bmich)', r'(?<!\bMICH)', r'(?<!\bMich)',
#                                 r'(?<!\bMinn)', r'(?<!\bminn)', r'(?<!\bMINN)', r'(?<!\bMinn)', r'(?<!\bMiss)', r'(?<!\bmiss)', r'(?<!\bMISS)',
#                                 r'(?<!\bMiss)', r'(?<!\bMmes)', r'(?<!\bmmes)', r'(?<!\bMMES)', r'(?<!\bMmes)', r'(?<!\bMo)', r'(?<!\bmo)', r'(?<!\bMO)',
#                                 r'(?<!\bMo)', r'(?<!\bMr)', r'(?<!\bmr)', r'(?<!\bMR)', r'(?<!\bMr)', r'(?<!\bMrs)', r'(?<!\bmrs)', r'(?<!\bMRS)',
#                                 r'(?<!\bMrs)', r'(?<!\bMs)', r'(?<!\bms)', r'(?<!\bMS)', r'(?<!\bMs)', r'(?<!\bMx)', r'(?<!\bmx)', r'(?<!\bMX)',
#                                 r'(?<!\bMx)', r'(?<!\bMt)', r'(?<!\bmt)', r'(?<!\bMT)', r'(?<!\bMt)', r'(?<!\bNO)', r'(?<!\bno)', r'(?<!\bNO)',
#                                 r'(?<!\bNo)', r'(?<!\bNo)', r'(?<!\bno)', r'(?<!\bNO)', r'(?<!\bNo)', r'(?<!\bNov)', r'(?<!\bnov)', r'(?<!\bNOV)',
#                                 r'(?<!\bNov)', r'(?<!\bOct)', r'(?<!\boct)', r'(?<!\bOCT)', r'(?<!\bOct)', r'(?<!\bOkla)',
#                                 r'(?<!\bokla)', r'(?<!\bOKLA)', r'(?<!\bOkla)', r'(?<!\bOp)', r'(?<!\bop)',
#                                 r'(?<!\bOP)', r'(?<!\bOp)', r'(?<!\bOre)', r'(?<!\bore)', r'(?<!\bORE)', r'(?<!\bOre)',
#                                 r'(?<!\bPa)', r'(?<!\bpa)', r'(?<!\bPA)', r'(?<!\bPa)', r'(?<!\bPp)', r'(?<!\bpp)',
#                                 r'(?<!\bPP)', r'(?<!\bPp)', r'(?<!\bProf)', r'(?<!\bprof)', r'(?<!\bPROF)', r'(?<!\bProf)',
#                                 r'(?<!\bProp)', r'(?<!\bprop)', r'(?<!\bPROP)', r'(?<!\bProp)', r'(?<!\bRd)', r'(?<!\brd)',
#                                 r'(?<!\bRD)', r'(?<!\bRd)', r'(?<!\bRef)', r'(?<!\bref)', r'(?<!\bREF)', r'(?<!\bRef)',
#                                 r'(?<!\bRep)', r'(?<!\brep)', r'(?<!\bREP)', r'(?<!\bRep)', r'(?<!\bReps)',
#                                 r'(?<!\breps)', r'(?<!\bREPS)', r'(?<!\bReps)', r'(?<!\bRev)', r'(?<!\brev)', r'(?<!\bREV)',
#                                 r'(?<!\bRev)', r'(?<!\bRte)', r'(?<!\brte)', r'(?<!\bRTE)', r'(?<!\bRte)', r'(?<!\bSen)',
#                                 r'(?<!\bsen)', r'(?<!\bSEN)', r'(?<!\bSen)', r'(?<!\bSept)', r'(?<!\bsept)',
#                                 r'(?<!\bSEPT)', r'(?<!\bSept)', r'(?<!\bSr)', r'(?<!\bsr)', r'(?<!\bSR)', r'(?<!\bSr)',
#                                 r'(?<!\bSt)', r'(?<!\bst)', r'(?<!\bST)', r'(?<!\bSt)', r'(?<!\bStat)', r'(?<!\bstat)',
#                                 r'(?<!\bSTAT)', r'(?<!\bStat)', r'(?<!\bSupt)', r'(?<!\bsupt)', r'(?<!\bSUPT)',
#                                 r'(?<!\bSupt)', r'(?<!\bTech)', r'(?<!\btech)', r'(?<!\bTECH)', r'(?<!\bTech)',
#                                 r'(?<!\bTex)', r'(?<!\btex)', r'(?<!\bTEX)', r'(?<!\bTex)', r'(?<!\bVa)',
#                                 r'(?<!\bva)', r'(?<!\bVA)', r'(?<!\bVa)', r'(?<!\bVol)', r'(?<!\bvol)', r'(?<!\bVOL)',
#                                 r'(?<!\bVol)', r'(?<!\bWash)', r'(?<!\bwash)', r'(?<!\bWASH)', r'(?<!\bWash)',
#                                 r'(?<!\bal)', r'(?<!\bal)', r'(?<!\bAL)', r'(?<!\bAl)', r'(?<!\bav)', r'(?<!\bav)',
#                                 r'(?<!\bAV)', r'(?<!\bAv)', r'(?<!\bave)', r'(?<!\bave)', r'(?<!\bAVE)', r'(?<!\bAve)',
#                                 r'(?<!\bca)', r'(?<!\bca)', r'(?<!\bCA)', r'(?<!\bCa)', r'(?<!\bcc)', r'(?<!\bcc)',
#                                 r'(?<!\bCC)', r'(?<!\bCc)', r'(?<!\bchap)', r'(?<!\bchap)', r'(?<!\bCHAP)', r'(?<!\bChap)',
#                                 r'(?<!\bcm)', r'(?<!\bcm)', r'(?<!\bCM)', r'(?<!\bCm)', r'(?<!\bcu)', r'(?<!\bcu)',
#                                 r'(?<!\bCU)', r'(?<!\bCu)', r'(?<!\bdia)', r'(?<!\bdia)', r'(?<!\bDIA)', r'(?<!\bDia)',
#                                 r'(?<!\bdr)', r'(?<!\bdr)', r'(?<!\bDR)', r'(?<!\bDr)', r'(?<!\beqn)', r'(?<!\beqn)',
#                                 r'(?<!\bEQN)', r'(?<!\bEqn)', r'(?<!\betc)', r'(?<!\betc)', r'(?<!\bETC)', r'(?<!\bEtc)',
#                                 r'(?<!\bfig)', r'(?<!\bfig)', r'(?<!\bFIG)', r'(?<!\bFig)', r'(?<!\bfigs)', r'(?<!\bfigs)',
#                                 r'(?<!\bFIGS)', r'(?<!\bFigs)', r'(?<!\bft)', r'(?<!\bft)', r'(?<!\bFT)', r'(?<!\bFt)',
#                                 r'(?<!\bgm)', r'(?<!\bgm)', r'(?<!\bGM)', r'(?<!\bGm)', r'(?<!\bhr)', r'(?<!\bhr)',
#                                 r'(?<!\bHR)', r'(?<!\bHr)', r'(?<!\bin)', r'(?<!\bin)', r'(?<!\bIN)', r'(?<!\bIn)',
#                                 r'(?<!\bkc)', r'(?<!\bkc)', r'(?<!\bKC)', r'(?<!\bKc)', r'(?<!\blb)', r'(?<!\blb)',
#                                 r'(?<!\bLB)', r'(?<!\bLb)', r'(?<!\blbs)', r'(?<!\blbs)', r'(?<!\bLBS)', r'(?<!\bLbs)',
#                                 r'(?<!\bmg)', r'(?<!\bmg)', r'(?<!\bMG)', r'(?<!\bMg)', r'(?<!\bml)', r'(?<!\bml)',
#                                 r'(?<!\bML)', r'(?<!\bMl)', r'(?<!\bmm)', r'(?<!\bmm)', r'(?<!\bMM)', r'(?<!\bMm)',
#                                 r'(?<!\bmv)', r'(?<!\bmv)', r'(?<!\bMV)', r'(?<!\bMv)', r'(?<!\bnw)', r'(?<!\bnw)',
#                                 r'(?<!\bNW)', r'(?<!\bNw)', r'(?<!\boz)', r'(?<!\boz)', r'(?<!\bOZ)', r'(?<!\bOz)',
#                                 r'(?<!\bpl)', r'(?<!\bpl)', r'(?<!\bPL)', r'(?<!\bPl)', r'(?<!\bpp)', r'(?<!\bpp)',
#                                 r'(?<!\bPP)', r'(?<!\bPp)', r'(?<!\bsec)', r'(?<!\bsec)', r'(?<!\bSEC)', r'(?<!\bSec)',
#                                 r'(?<!\bsq)', r'(?<!\bsq)', r'(?<!\bSQ)', r'(?<!\bSq)', r'(?<!\bst)', r'(?<!\bst)',
#                                 r'(?<!\bST)', r'(?<!\bSt)', r'(?<!\bvs)', r'(?<!\bvs)', r'(?<!\bVS)', r'(?<!\bVs)',
#                                 r'(?<!\byr)', r'(?<!\byr)', r'(?<!\bYR)', r'(?<!\bYr)']


def preproc1(comment, steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment

    comp = functools.partial(re.compile, flags=re.IGNORECASE)
    if 1 in steps:  # replace newlines with spaces
        # modComm = re.sub(r"\n{1,}", " ", modComm)
        modComm = modComm.replace('\n', '').replace('\r', '').replace('\t', '')
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces and punctuation
        # punctuation = r"[!\"\\#$%&()*+,\-\./:;<=>?@\[\\\]^_`{|}~]"
        # abbrevs = "".join(common_abbreviations)
        # regex = rf"((?" \
        #   rf":(?<!\w\.\w){abbrevs}){punctuation}+(?!\w\.))"

        # modComm = re.sub(regex, r' \1 ', modComm)
        modComm = re.sub(r'( {2,})', '', modComm)
        modComm = re.sub(r"(\s*$)", "", modComm)
    if modComm == '':
        return '\n'
    utterance = nlp(modComm)
    modComm = ''
    for sent in utterance.sents:
        # if 5 in steps:  # remove contractions
        #     # patterns = [(r"(\'(?=(ve|d|ll|n|re|s|m)))", r' \1'),
        #     #             (r"((?<=t|y)\')", r' \1'),
        #     #             (r"(n\'t)", r' \1'),
        #     #             (r"(s\')", r"s '")]
        #     patterns = [(r"(\'ve)", 'have'),
        #                 (r"(\'d)", 'did'),
        #                 (r"(\'ll)", 'will'),
        #                 (r"(\'re)", 'are'),
        #                 (r"(\'s)", 'is'),
        #                 (r"(\'m)", 'be'),
        #                 (r"(n\'t)", 'not'),
        #                 (r"(s\')", "s '")]
        #     for pattern, repl in patterns:
        #         modComm = comp(pattern).sub(' '+repl, modComm)

        if 6 in steps:  # POS tagging, stop word removal
            def new_str(pos):
                for token in pos:
                    if token.lemma_[0] == '-' and token.text[0] != '-':
                        beg = token.text
                    else:
                        beg = token.lemma_
                    yield beg + '/' + token.tag_
            # modComm = " ".join([token.text+"/"+token.tag_ for token in tags if not token.is_stop])
            modComm += " ".join(new_str(sent)) + '\n'
            # modComm = re.sub(r'\s\n', "\n", modComm, re.IGNORECASE)

        # if 7 in steps:  # Lemmatization
        #     modComm += " "
        #     # Remove tags
        #     print(re.findall(r"(/\S*\s)", modComm))
        #     modComm = re.sub(r"(/\S*\s)", " ", modComm)
        #     pos = nlp(modComm)
        #     def new_str(pos):
        #         for token in pos:
        #             if token.lemma_[0] == '-' and token.text[0] != '-':
        #                 beg = token.text
        #             else:
        #                 beg = token.lemma_
        #             yield beg + '/' + token.tag_
        #     modComm = " ".join(new_str(pos))

        # if 9 in steps:  # Add new lines between sentences
        #     # assemble abbreviations
        #     # abbrevs = "".join(common_abbreviations)
        #     punctuation = r"([\.?!]|(\".*\"))\s"
        #     # regex = rf"(((?<!\w\.\w){abbrevs}){punctuation})
        #     regex = rf"((?<!\w\.\w){punctuation})"
        #     modComm = re.sub(r"( {2,})", ' ', re.sub(regex, r'\1\n', modComm))
        #     modComm = re.sub(r"(\s*$)", r"\n", modComm)

        # if 10 in steps:
        #     modComm = modComm.lower()

        # TODO: get Spacy document for modComm

        # TODO: use Spacy document for modComm to create a string.
        # Make sure to:
        #    * Insert "\n" between sentences.
        #    * Split tokens with spaces.
        #    * Write "/POS" after each token.
        
    return modComm


def main(args):
    allOutput = []
    indir = args.a1_dir
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            if file not in ['Center', 'Right', 'Left', 'Alt']:
                continue
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            sindex = args.ID[0] % len(data)
            stime = time.time()
            for i in range(sindex, sindex + args.max):
            # for i in range(0, 5):
                if i > len(data):  # Ensure circular indexing
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
