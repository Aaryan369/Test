import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

def read_glove_vecs(glove_file):
    with open(glove_file, encoding = "utf-8") as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            words.add(line[0])
            word_to_vec_map[line[0]] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

def cosine_similarity(u, v):
    dot = np.dot(u,v)

    norm_u = np.sqrt(np.sum(u*u))
    norm_v = np.sqrt(np.sum(v*v))

    cosine_similarity = dot/(norm_u*norm_v)
   
    return cosine_similarity

def complete_analogy(A, B, C, words, word_to_vec_map):
    A, B, C = A.lower(), B.lower(), C.lower()
    e_a, e_b, e_c = word_to_vec_map[A], word_to_vec_map[B], word_to_vec_map[C]

    max_cosine_sim = float('-inf')             
    best_word = None                   
    input_words_set = set([A, B, C])
    for w in words:        
        if w in input_words_set:
            continue
        cosine_sim = cosine_similarity(e_b - e_a,word_to_vec_map[w]- e_c)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w   
    return best_word

words, word_to_vec_map = read_glove_vecs('glove.txt')
#print(complete_analogy('man', 'woman', 'boy',words ,word_to_vec_map))

@app.route("/")
def index():
    return render_template("question.html")

@app.route("/answer",methods=['POST'])
def answer():
    A = request.form['A']
    B = request.form['B']
    C = request.form['C']
    D = complete_analogy(A, B, C,words ,word_to_vec_map)
    return render_template("answers.html",A=A.title(),B=B.title(),C=C.title(),D=D.title())
    

if __name__ == '__main__':
   app.run(port=3000, debug = True)