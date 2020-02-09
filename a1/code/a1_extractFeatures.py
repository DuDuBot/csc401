import numpy as np
import argparse
import json
import pandas as pd
import re
import functools
import os
import time

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
  'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
  'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
  'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
  'their', 'theirs'}
SLANG = {
  'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
  'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
  'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
  'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
  'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}


def len_dec(fn):
  def len_fn(*args, **kwargs):
    return len(fn(*args, **kwargs))

  return len_fn



# Global vars
findall = functools.partial(re.findall, flags=re.IGNORECASE)
nfindall = len_dec(findall)  # nfindall return number of matches from findall

a1_dir = "/u/cs401/A1/"  # can't acess args...
wordlists_dir = "/u/cs401/Wordlists"
bgl = pd.read_csv(os.path.join(wordlists_dir, "BristolNorms+GilhoolyLogie.csv"))
warr = pd.read_csv(os.path.join(wordlists_dir, "Ratings_Warriner_et_al.csv"))

bgl = bgl[["WORD", "AoA (100-700)", "IMG", "FAM"]]  # select just needed cols
warr = warr[["Word", "V.Mean.Sum", "D.Mean.Sum", "A.Mean.Sum"]]
bgl2 = bgl.set_index(['WORD'])
warr2 = warr.set_index(['Word'])

feats_base = os.path.join(a1_dir, 'feats')
files = ['Left', 'Center', 'Right', 'Alt']  # files for different labels, in order.
files = {f: [open(os.path.join(feats_base, f + '_IDs.txt')).readlines(),  # [0] is the IDs
             np.load(os.path.join(feats_base, f + '_feats.dat.npy')),  # [1] is the features
             i] for i, f in enumerate(files)  # [2] is the integer label
         }

for v in files.values():
    # remove the [2] entry and concatenate to features as desired.
    v[1] = np.concatenate([v[1], np.ones((len(v[1]), 1)) * v.pop()], axis=1)

# make dict for id : line
for k in files:
    id_line = {}
    for i, line in enumerate(files[k][0]):
        id_line[str(line).strip()] = i
    files[k][0] = id_line


def extract1(comment):
  ''' This function extracts features from a single comment

  Parameters:
      comment : string, the body of a comment (after preprocessing)
      # bgl: pandas DataFrame of Bristol Gillhooly and Logie features. (to avoid
      #     needing to reopen every time)
      # warr: pandas DataFrame of Warringer features. (to avoid needing to
      #     reopen every time)

  Returns:
      feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
  '''
  def flatten(arr):
    new_arr = []
    for a in arr:
      if isinstance(a, pd.Series):
        new_arr.extend(a.values)
      else:
        new_arr.append(a)
    return new_arr
  features = np.zeros(173)
  patterns = [(r"([A-Z]\w{2,})/[A-Z]{2,4}", {'flags': 0}),  # 1. n uppercase > 3
              (r"(?<=\b)(?:" + "|".join(FIRST_PERSON_PRONOUNS) + ")(?=\/|\b)",
               {}),  # 2. n first-person pronouns
              (r"(?<=\b)(?:" + "|".join(SECOND_PERSON_PRONOUNS) + ")(?=\b|\/)",
               {}),  # 3. n second-person pronouns
              (r"(?<=\b)(?:" + "|".join(THIRD_PERSON_PRONOUNS) + ")(?=\b|\/)",
               {}),  # 4. n third-person pronouns
              (r"(/CC)(?=\b)", {}),  # 5. n coordinating conjunctions
              (r"(/VBD)(?=\b)", {}),  # 6. n past-tense verbs
              (
              r"(?:(?:going|gonna|will|'ll)(?:/?.{0,4})|go/VBG)\s+to(?:/.{0,2})\s+(\w*/VB)(?=\b|/)",
              {}),  # 7. n future tense, .{} are for tags.
              (r",/,|(?<!/),", {}),  # 8. n commas
              #(r"[`~!@#$%\^&\*\(\):;\"\'\\\+,\-\.<>=?\[\]_\{\|\}']{2,}", {}),
              (r"([\#\$\!\?\.\:\;\(\)\"\',]{2,}|\.\.\.)", {}),
              # 9. n multchar punctuations
              (r"(?:/NN|/NNS)(?=\b)", {}),  # 10. n common nouns
              (r"(?:/NNP|/NNPS)(?=\b)", {}),  # 11.n proper nouns
              (r"(?:/RB|/RBR|/RBS)(?=\b)", {}),  # 12. n adverbs
              (r"(?:/WDT\b|/WP\$\W|/WRB\b|/WP\b)", {}),  # 13. n wh-words
              # (r"(\b"+r"|".join(SLANG)+")(?:/[.]{0,4})", {}),
              (r"(?:\s|^)("+r"|".join(SLANG)+")(?:/[.]{0,4})", {})  # 14. n slang words
              ]
  word_tokens = comment.split()  # as per preprocessing
  # words, tokens = [], []
  # for string in word_tokens:
  #   words.append(string[:string.rfind('/')])
  #   tokens.append(string[string.rfind('/')+1:])
  # for word, token in zip(words, tokens):
  #     features[0] += nfindall(r"([A-Z]\w{2,})", word, flags=0)  # 1. n uppercase > 3
  #     features[1] += 1 if word in FIRST_PERSON_PRONOUNS else 0  # 2. n first-person pronouns
  #     features[2] += 1 if word in SECOND_PERSON_PRONOUNS else 0  # 3. n second-person pronouns
  #     features[3] += 1 if word in THIRD_PERSON_PRONOUNS else 0  # 4. n third-person pronouns
  #     if token == 'CC':
  #         features[4] += 1  # 5. n coordinating conjunctions
  #     elif token == 'VBD':
  #         features[5] += 1  # 6. n past-tense verbs
  #     elif token == 'NN' or token == 'NNS':
  #         features[9] += 1  # 10. n common nouns
  #     elif token == 'NNP' or token == 'NNPS':
  #         features[10] += 1  # 11. n proper nouns
  #     elif token == 'RB' or token == 'RBR' or token == 'RBS':
  #         features[11] += 1  # 12. n adverbs
  #     elif token == 'WDT' or token == 'WP$' or token == 'WRB' or token == 'WP':
  #         features[12] += 1  # 13. n wh-words
  #     features[7] += nfindall(r",/,|(?<!/),", string)  # 8. n commas
  #     features[8] += nfindall(r"([\#\$\!\?\.\:\;\(\)\"\',]{2,}|\.\.\.)", word)  # 9. n multchar punctuations
  #     features[13] += 1 if word in SLANG else 0  # 14. n slang words
  # features[6] += nfindall(r"(?:(?:going|gonna|will|'ll)"
  #                           r"(?:/?.{0,4})|go/VBG)\s+to"
  #                           r"(?:/.{0,2})\s+(\w*/VB)(?=\b|/)", comment)  # 7. n future tense
  for i, (patt, flags) in enumerate(patterns):
    try:
      if len(flags) > 0:
        # nfindall return number of matches from findall
        features[i] += nfindall(patt, comment, **flags)
      else:
        # nfindall return number of matches from findall
        features[i] += nfindall(patt, comment)
    except:
      import traceback
      print(traceback.format_exc())

  check_punct = r"([\#\$\!\?\.\:\;\(\)\"\',\[\]/]{1,}|\.\.\.)"  # check for punctuation
  sentences = comment.split("\n")[:-1]  # as per processing, we have extra \n at end
  if len(word_tokens) > 0:  # extract just word from each word/token pair
    retrieve_word = r"(/?\w+)(?=/)"  # they are separated by a / with token
    extract_words = [findall(retrieve_word, word) for word in word_tokens]
    extract_words = [w[0].lower() for w in extract_words if
                     len(w) > 0]  # they are lists from findall
  if len(sentences) > 0:  # check divide by 0 errors.
    # 15. Avg n tokens in sentence
    # nfindall return number of matches from findall
    features[14] = sum(
      [nfindall(r"(\b/.{0,4}(?:\s|$)|"+f"{check_punct}/"+r"[\-]?.{0,4}[\-]?(?:\s|$))", s) for s in sentences]) / len(
      sentences)
    # 17. n sentences
    features[16] = len(sentences)

  if len(word_tokens) > 0:
    # 16. Avg len tokens excluding punctutation only, in chars
    # findall returns all matches
    # below returns None for tokens proceeded by only punctuation.
    valid_tokens = [w for w in word_tokens if re.match(rf"{check_punct}/", w) is None]
    # print([v[:v.rfind('/')] for v in valid_tokens])
    if len(extract_words) > 0:
      features[15] = len("".join([v[:v.rfind('/')] for v in valid_tokens])) / (len(
        valid_tokens))  # n chars / n tokens
    # print(features[15])

  # Norms
  if len(word_tokens) > 0:  # extract just word from each word
    # retrieve_word = r"(\w+)(?=/)"  # they are separated by a / with token
    # extract_words = [findall(retrieve_word, word) for word in words]
    # extract_words = [w[0].lower() for w in extract_words if
    #                  len(w) > 0]  # they are lists from findall
    # valid_bgl = bgl2.loc[np.in1d(bgl2.index, extract_words)]  # words from comment in bgl
    chosen_bgl = []
    for x in extract_words:
      try:
        chosen_bgl.append(bgl2.loc[x])  # some words might not have a value
      except:
        pass
    # Below are all features Bristol Gillhooly and Logie
    # Start with AoA
    AoA = [x.get("AoA (100-700)", np.nan) for x in chosen_bgl]
    AoA = flatten(AoA)
    # Do the two BGL, AoA cases
    if np.count_nonzero(~np.isnan(AoA)) > 0:
      # 18. norms average AoA
      features[17] = np.nanmean(AoA)
      # 21. standard deviation AoA
      features[20] = np.nanstd(AoA)
    # now IMG
    IMG = [x.get("IMG", np.nan) for x in chosen_bgl]
    IMG = flatten(IMG)
    if np.count_nonzero(~np.isnan(IMG)) > 0:
      # 19. average IMG
      features[18] = np.nanmean(IMG)
      # 22. standard deviation IMG
      features[21] = np.nanstd(IMG)

    # finally FAM
    FAM = [x.get("FAM", np.nan) for x in chosen_bgl]
    FAM = flatten(FAM)
    if np.count_nonzero(~np.isnan(FAM)) > 0:
      # 20. average FAM
      features[19] = np.nanmean(FAM)
      # 23. standard deviation FAM
      features[22] = np.nanstd(FAM)

    # Now we start Warringer norms
    # valid_warr = [warr[warr.Word == w] for w in extract_words]  # words from comment in warr
    # DMS = [w["V.Mean.Sum"].values[0] for w in valid_warr if not w["V.Mean.Sum"].empty]
    # valid_warr = warr2.loc[np.in1d(warr2.index, extract_words)]  # words from comment in warr
    chosen_warr = []
    for x in extract_words:
      try:
        chosen_warr.append(warr2.loc[x])  # some words might not have a value
      except:
        pass
    # VMS = [w["V.Mean.Sum"].values[0] for w in valid_warr if not w["V.Mean.Sum"].empty]  # V.Mean.Sum from war
    # first V.Mean.Sum
    VMS = [x.get("V.Mean.Sum", np.nan) for x in chosen_warr]
    VMS = flatten(VMS)
    if np.count_nonzero(~np.isnan(VMS)) > 0:
      # 24. average V.Mean.Sum
      features[23] = np.nanmean(VMS)
      # 27. standard deviation V.Mean.Sum
      features[26] = np.nanstd(VMS)
    # second A.Mean.Sum
    AMS = [x.get("A.Mean.Sum", np.nan) for x in chosen_warr]
    AMS = flatten(AMS)
    if np.count_nonzero(~np.isnan(AMS)) > 0:
      # 25. average A.Mean.Sum
      features[24] = np.nanmean(AMS)
      # 28. standard deviation A.Mean.Sum
      features[27] = np.nanstd(AMS)
    # third D.Mean.Sum
    DMS = [x.get("D.Mean.Sum", np.nan) for x in chosen_warr]
    DMS = flatten(DMS)
    if np.count_nonzero(~np.isnan(DMS)) > 0:
      # 26. average D.Mean.Sum
      features[25] = np.nanmean(DMS)
      # 29. standard deviation D.Mean.Sum
      features[28] = np.nanstd(DMS)

  return features


def extract2(feats, comment_class, comment_id):
  ''' This function adds features 30-173 for a single comment.

  Parameters:
      feats: np.array of length 173
      comment_class: str in {"Alt", "Center", "Left", "Right"}
      comment_id: int indicating the id of a comment
      # files: dict of category to opened files (to avoid needing to reopen
      #     every time) and to the features

  Returns:
      feats : numpy Array, a 173-length vector of floating point features (this
      function adds feature 30-173). This should be a modified version of
      the parameter feats.
  '''
  class_file = files[comment_class]
  if comment_id in class_file[0]:
    feats[29:] = class_file[1][class_file[0][comment_id], :-1]
  else:
    print('not found')
  return feats


def main(args):
  # Now, files is a mapping from the {label: [ids, features]},
  # where features has the label integer concatenated to the end.

  data = json.load(open(args.input))
  feats = np.zeros((len(data), 173 + 1))

  stime = time.clock()
  for i, comment in enumerate(data):
    if (i+1) % 500 == 0:
      print(f"step: '{i+1}' at time '{time.clock()-stime}'")

    feats[i, :-1] = extract1(comment['body'])
    feats[i, :-1] = extract2(feats[i, :-1], comment['cat'], comment['id'])
    class_file = files[comment['cat']]
    feats[-1] = class_file[1][0, -1]

  np.savez_compressed(args.output, feats)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process each .')
  parser.add_argument("-o", "--output",
                      help="Directs the output to a filename of your choice",
                      required=True)
  parser.add_argument("-i", "--input",
                      help="The input JSON file, preprocessed as in Task 1",
                      required=True)
  parser.add_argument("-p", "--a1_dir",
                      help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.",
                      default="/u/cs401/A1/")
  args = parser.parse_args()

  main(args)
