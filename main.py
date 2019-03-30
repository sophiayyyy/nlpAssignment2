import os
import csv
import subprocess
import re
import random
import numpy as np
import math
from scipy import sparse


def read_in_shakespeare():
  '''Reads in the Shakespeare dataset processesit into a list of tuples.
     Also reads in the vocab and play name lists from files.

  Each tuple consists of
  tuple[0]: The name of the play
  tuple[1] A line from the play as a list of tokenized words.

  Returns:
    tuples: A list of tuples in the above format.
    document_names: A list of the plays present in the corpus.
    vocab: A list of all tokens in the vocabulary.
  '''

  tuples = []
  character_names = []
  with open('will_play_text.csv') as f:
    csv_reader = csv.reader(f, delimiter=';')
    for row in csv_reader:
      play_name = row[1]
      character_name = row[4]
      line = row[5]
      line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
      line_tokens = [token.lower() for token in line_tokens]

      tuples.append((play_name, line_tokens, character_name))
      character_names.append(character_name)

  with open('vocab.txt') as f:
    vocab =  [line.strip() for line in f]

  with open('play_names.txt') as f:
    document_names =  [line.strip() for line in f]

  character_names_final = np.unique(character_names)

  return tuples, document_names, vocab, character_names_final

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
  '''Returns a numpy array containing the term document matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary
    
  Let m = len(vocab) and n = len(document_names).

  Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of words
        and each column corresponds to a document. A_ij contains the
        frequency with which word i occurs in document j.
  '''

  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  docname_to_id = dict(zip(document_names, range(0, len(document_names))))
  # YOUR CODE HERE
  m = len(vocab)
  n = len(document_names)
  document_matrix = np.zeros(shape=(m, n))
  for l in line_tuples:
    d = docname_to_id.get(l[0], None)
    if d is not None:
      for word in l[1]:
        v = vocab_to_id.get(word, None)
        if v is not None:
          document_matrix[v][d] += 1
  return document_matrix

def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
  '''Returns a numpy array containing the term context matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    vocab: A list of the tokens in the vocabulary

  Let n = len(vocab).

  Returns:
    tc_matrix: A nxn numpy array where A_ij contains the frequency with which
        word j was found within context_window_size to the left or right of
        word i in any sentence in the tuples.
  '''
  # YOUR CODE HERE
  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  n = len(vocab)
  context_matrix = np.zeros(shape=(n, n))
  for l in line_tuples:
    for i in range(len(l[1])):
      left = i - context_window_size
      right = i + context_window_size
      if left < 0:
        left = 0
      if right >= len(l[1]):
        right = len(l[1]) - 1
      for j in range(left, right + 1):
        v1 = vocab_to_id.get(l[1][i], None)
        v2 = vocab_to_id.get(l[1][j], None)
        if v1 is not None and v2 is not None:
          context_matrix[v1][v2] += 1
  return context_matrix

def create_PPMI_matrix(term_context_matrix):
  '''Given a term context matrix, output a PPMI matrix.
    
  Hint: Use numpy matrix and vector operations to speed up implementation.
  
  Input:
    term_context_matrix: A nxn numpy array, where n is
        the numer of tokens in the vocab.
  
  Returns: A nxn numpy matrix, where A_ij is equal to the
     point-wise mutual information between the ith word
     and the jth word in the term_context_matrix.
  '''       
  
  # YOUR CODE HERE
  n = term_context_matrix.shape[0]
  PPMI_matrix = np.zeros(shape=(n, n))
  matrix_sum = np.sum(term_context_matrix) + 2 * pow(n, 2)
  matrix_row_sum = np.sum(term_context_matrix,axis=1)
  matrix_col_sum = np.sum(term_context_matrix,axis=0)
  for i in range(n):
    for j in range(n):
      pij = term_context_matrix[i][j] + 2
      pi = matrix_row_sum[i] + 2 * n
      pj = matrix_col_sum[j] + 2 * n
      if matrix_row_sum[i] == 0 or matrix_col_sum[j] == 0 or pij == 0:
        PPMI_matrix[i][j] = 0
      else:
        PPMI_matrix[i][j] = (float(pij) * matrix_sum)/(pi * pj)
  tmp = np.log2(PPMI_matrix)
  PPMI_matrix_final = np.maximum(tmp, 0)
  print(PPMI_matrix_final)
  data = sparse.csr_matrix(PPMI_matrix_final)
  # Save
  sparse.save_npz('ppmi.npz', data)
  return PPMI_matrix_final

def create_tf_idf_matrix(term_document_matrix):
  '''Given the term document matrix, output a tf-idf weighted version.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.

  Input:
    term_document_matrix: Numpy array where each column represents a document 
    and each row, the frequency of a word in that document.

  Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
  '''

  # YOUR CODE HERE
  m = term_document_matrix.shape[0]
  n = term_document_matrix.shape[1]
  tf_idf_matrix = np.zeros(shape=(m, n))
  for i in range(m):
    dfi = 0
    for j in range(n):
      if term_document_matrix[i][j] != 0:
        dfi += 1
    idf = math.log(float(n) / dfi)
    for j in range(n):
      if term_document_matrix[i][j] == 0:
        tf = 0
      else:
        tf = 1 + math.log10(term_document_matrix[i][j])
      tf_idf_matrix[i][j] = tf * idf
  return tf_idf_matrix

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
  tmp = np.linalg.norm(vector1) * np.linalg.norm(vector2)
  if tmp == 0:
    value = 0
  else:
    value = np.dot(vector1, vector2) / tmp
  return value

def compute_jaccard_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
  # Jaccard similarity can be measured using your vectors e.g., a and b as: sum(min(a, b))/sum(max(a,b))
  maxsum = math.fsum(np.maximum(vector1, vector2))
  minsum = math.fsum(np.minimum(vector1, vector2))
  return float(minsum) / maxsum

def compute_dice_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''

  # YOUR CODE HERE
  # Dice can be measured using your vectors a and b as 2 * sum(min(a,b)) / sum(a+b)
  minsum = math.fsum(np.minimum(vector1, vector2))
  sum = math.fsum(np.concatenate((vector1, vector2),axis=0))
  return float(minsum) * 2 / sum

def rank_plays(target_play_index, term_document_matrix, similarity_fn):
  ''' Ranks the similarity of all of the plays to the target play.

  Inputs:
    target_play_index: The integer index of the play we want to compare all others against.
    term_document_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index
  '''
  
  # YOUR CODE HERE
  n = term_document_matrix.shape[1]
  value = np.zeros(n)
  v1 = term_document_matrix[:, target_play_index]
  for i in range(n):
    if target_play_index == i:
      value[i] = 0
      continue
    v2 = term_document_matrix[:, i]
    value[i] = similarity_fn(v1, v2)
  return np.argsort(-value)

def rank_words(target_word_index, matrix, similarity_fn):
  ''' Ranks the similarity of all of the words to the target word.

  Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
    similarity_fn: Function that should be used to compared vectors for two word
      embeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the 
    target word indexed by word_index
  '''

  # YOUR CODE HERE
  n = matrix.shape[0]
  value = np.zeros(n)
  v1 = matrix[target_word_index, :]
  for i in range(n):
    if target_word_index == i:
      value[i] = 0
      continue
    v2 = matrix[i, :]
    value[i] = similarity_fn(v1, v2)
  return np.argsort(-value)

def create_term_character_matrix(line_tuples, character_names, vocab):
  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  chaname_to_id = dict(zip(character_names, range(0, len(character_names))))
  # YOUR CODE HERE
  m = len(vocab)
  n = len(character_names)
  character_matrix = np.zeros(shape=(m, n))
  for l in line_tuples:
    c = chaname_to_id.get(l[2], None)
    if c is not None:
      for word in l[1]:
        v = vocab_to_id.get(word, None)
        if v is not None:
          character_matrix[v][c] += 1
  return character_matrix


def rank_characters(target_character_index, term_character_matrix, similarity_fn):
  ''' Ranks the similarity of all of the characters to the target character.

  Inputs:
    target_character_index: The integer index of the character we want to compare all others against.
    term_character_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index
  '''
  # YOUR CODE HERE
  n = term_character_matrix.shape[1]
  value = np.zeros(n)
  v1 = term_character_matrix[:, target_character_index]
  for i in range(n):
    if target_character_index == i:
      value[i] = 0
      continue
    v2 = term_character_matrix[:, i]
    value[i] = similarity_fn(v1, v2)
  return np.argsort(-value)

def cluster_document(term_character_matrix, document_names):
  from sklearn.cluster import KMeans
  X = np.array(term_character_matrix.T)
  kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
  labels = kmeans.labels_
  cluster_first = np.where(labels == 0)[0]
  cluster_second = np.where(labels == 1)[0]
  cluster_third = np.where(labels == 2)[0]

  id_to_docname = dict(zip(range(0, len(document_names)), document_names))
  print('Cluster One:')
  for l in cluster_first:
    d = id_to_docname.get(l, None)
    if d is not None:
      print(d)

  print('Cluster Two:')
  for l in cluster_second:
    d = id_to_docname.get(l, None)
    if d is not None:
      print(d)

  print('Cluster Three:')
  for l in cluster_third:
    d = id_to_docname.get(l, None)
    if d is not None:
      print(d)

if __name__ == '__main__':
  tuples, document_names, vocab, character_names = read_in_shakespeare()
  print('len of character_names', len(character_names))

  # term document matrix
  print('Computing term document matrix...')
  td_matrix = create_term_document_matrix(tuples, document_names, vocab)
  print('td_matrix is :', td_matrix)

  # term context matrix
  print('Computing term context matrix...')
  tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)
  print('tc_matrix is :', tc_matrix)

  # tf-idf matrix
  print('Computing tf-idf matrix...')
  tf_idf_matrix = create_tf_idf_matrix(td_matrix)
  print('tf_idf_matrix is :', tc_matrix)

  #PPMI matrix
  # print('Computing PPMI matrix...')
  # PPMI_matrix = create_PPMI_matrix(tc_matrix)

  #term character matrix
  print('Computing term character matrix...')
  tcha_matrix = create_term_character_matrix(tuples, character_names, vocab)
  print('tcha_matrix is :', tcha_matrix)

  #cluster the document into three categories
  cluster_document(td_matrix, document_names)

  #rank plays
  random_idx = random.randint(0, len(document_names)-1)
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
    print('\nThe top most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
    ranks = rank_plays(random_idx, td_matrix, sim_fn)
    for idx in range(0, 1):
      doc_id = ranks[idx]
      print('%d: %s' % (idx+1, document_names[doc_id]))

  #rank character
  random_idx = random.randint(0, len(character_names) - 1)
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
    print('\nThe top most similar character to "%s" using %s are:' % (character_names[random_idx], sim_fn.__qualname__))
    ranks = rank_plays(random_idx, tcha_matrix, sim_fn)
    for idx in range(0, 1):
      cha_id = ranks[idx]
      print('%d: %s' % (idx + 1, character_names[cha_id]))
    for idx in range(len(ranks) - 1, len(ranks)):
      cha_id = ranks[idx]
      print('%d: %s' % (idx + 1, character_names[cha_id]))

  #load PPMI_matrix form the npz file
  PPMI_matrix_final = sparse.load_npz("ppmi.npz").toarray()

  #rank words
  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on term-document frequency matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], td_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on tf_idf_matrix matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], tf_idf_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], PPMI_matrix_final, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))
