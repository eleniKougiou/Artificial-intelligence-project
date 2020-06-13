import os
import math
from nltk.corpus import stopwords 
import nltk
nltk.download('stopwords')
from itertools import islice 
from Node import Node



def entropy(c0, c1):
    return -1 * (c0 * log(c0) + c1* log(c1))


def decide(p_h, p_s): 
    if (p_h > p_s):
        return 0
    return 1
        
def log(number):
    if(number <= 0): 
        return 0
    else:
        return math.log(number, 2)

def create_instances(ham_train, spam_train, words):
    #ham = 0, spam = 1
    instances = []
    found = 0
    for ham in ham_train:
        #print(ham)
        i = [];
        i.append(0)
        for w in words:
            if ham.find(w) != -1:
                found += 1
                i.append(1)
            else:
                i.append(0)
        instances.append(i)
    for spam in spam_train:
        i = [];
        i.append(1)
        for w in words:
            if spam.find(w) != -1:
                found += 1
                i.append(1)
            else:
                i.append(0)
        instances.append(i)
    return instances
    

def prob(instance, x, c):
    s = 0
    a = 0
    k = 0
    for i in instance:
        if len(i) != 0:
            if i[0] == c:
                a += 1
                s += i[x]
            k += 1
    return (s + 1) / (a + 2)

           
def test_naive_bayes(ham_test, spam_test, c0, c1, words, hp, sp):
    instance = create_instances(ham_test, spam_test, words)
    it_was_ham = 0
    it_was_spam = 0
    count = [0, 0] #count ham, spam
    for i in instance:
        #ham
        p_c0 = hp 
        #spam
        p_c1 = sp
        for a in range(1000):
            if (i[a + 1] == 1):
                p_c0 *= c0[a]
                p_c1 *= c1[a]
            else:
                p_c0 *= (1 - c0[a])
                p_c1 *= (1 - c1[a])
                
        
        choose = decide(p_c0, p_c1)
        count[choose] += 1 
        if ( choose != i[0]): #if the prediction is different from the truth
            if (i[0] == 1):
                it_was_spam += 1
            else:
                it_was_ham += 1
    
    acc = accuracy((it_was_spam + it_was_ham), len(instance))
    precision_h = (count[0] - it_was_spam) / count[0] #how many of what were considered hams were indeed
    precision_s = (count[1] - it_was_ham) / count[1]
    recall_h = (count[0] - it_was_spam) / len(ham_test) #how many hams were found
    recall_s = (count[1] - it_was_ham) / len(spam_test) 
    f1_h = 2 * precision_h * recall_h / (precision_h + recall_h)
    f1_s = 2 * precision_s * recall_s / (precision_s + recall_s)
    return [acc, precision_h * 100, precision_s * 100, recall_h * 100, recall_s * 100, f1_h * 100, f1_s * 100]
    
def test_id3(trained_tree, ham_test, spam_test, words):
    instance = create_instances(ham_test, spam_test, words)
    it_was_ham = 0
    it_was_spam = 0
    count = [0, 0] #count [ham, spam]
    for i in instance:
        choose = predict(trained_tree, i)
        count[choose] += 1
        if ( choose != i[0]):
            if (i[0] == 1):
                it_was_spam += 1
            else:
                it_was_ham += 1
    
    acc = accuracy((it_was_spam + it_was_ham), len(instance))
    precision_h = (count[0] - it_was_spam) / count[0]
    precision_s = (count[1] - it_was_ham) / count[1]
    recall_h = (count[0] - it_was_spam) / len(ham_test)
    recall_s = (count[1] - it_was_ham) / len(spam_test)  
    return [acc, precision_h * 100, precision_s * 100, recall_h * 100, recall_s * 100]
            

def predict(tree, instance):
    if len(tree.children) == 0:
        #it is a leaf
        return tree.category
    else:
        attr_value = instance[tree.attribute]
        if attr_value in tree.children: 
            return predict(tree.children[attr_value], instance)
        else:
            instances = []
            for attr_value in range(2):
                instances += tree.children[attr_value].instances_labeled
            return find_class(instances)
    


def id3(instances, words, default):
    if len(instances) == 0:
    #examples are over, so return the default category
        return Node(default)
    s = 0
    for i in instances:
        s += i[0] #sum spam
    if ((s > (len(instances) * 0.95)) or (s < (len(instances) * 0.05))):
    #spams or hams exceed the 95% of the total data, so stop
    #premature termination 
        tree = Node(find_class(instances))
        return tree
    elif (len(words) == 0):
    #the properties are over, so return the most common category
        tree = Node(find_class(instances))
        return tree
    else:
        best_attr = best_attribute(instances, words)
        del words[best_attr] # remove the best_attr from words, so it is not used again
        tree = Node(find_class(instances))
        tree.attribute = best_attr
 
        for best_attr_value in range(2): 
        # for best_attr_value = 0 or 1 
            instances_i = []
            for instance in instances:
                if instance[best_attr] == best_attr_value:
                    instances_i.append(instance)
            subtree = id3(instances_i, words, find_class(instances))
            subtree.instances_labeled = instances_i
            subtree.parent_attribute = best_attr # parent node
            subtree.parent_attribute_value = best_attr_value
            tree.children[best_attr_value] = subtree
        return tree
    
def find_class(instance):
    #return the most common category
    s = 0 
    for i in instance:
        s += i[0] #sum spam
    if s > (len(instance) - s): # if spam more than ham
    return 0


def best_attribute(instance, words):
    s = 0
    for i in instance:
        s += i[0]
    spam_prop = s / len(instance)
    ham_prop = 1 - spam_prop
    H = -1 * spam_prop * math.log(spam_prop, 2) - ham_prop * math.log(ham_prop, 2) 
    max_ig = -1
    best_attr = 0
    
    for a, word in enumerate(words):

        count_ham = 0
        count_spam = 0
        count_ham_yparxei = 0
        count_spam_yparxei = 0
        for i in instance:
            if (i[0] == 0): #ham
                count_ham += 1
                count_ham_yparxei += i[a + 1]
            else:
                count_spam +=1
                count_spam_yparxei += i[a + 1]
        synolo = len(instance)
        if (synolo == 0):
            print("oh")
        prob_yparxei = (count_ham_yparxei + count_spam_yparxei) / synolo
        prob_d_yparxei = 1 - prob_yparxei
        if ((count_ham_yparxei + count_spam_yparxei) == 0):
        #if there are no data, the probability is 50%
	    prob_ham_if_yparxei = 0.5
        else:
            prob_ham_if_yparxei = count_ham_yparxei / (count_ham_yparxei + count_spam_yparxei)
        prob_spam_if_yparxei = 1 - prob_ham_if_yparxei
        h_if_yparxei = entropy(prob_ham_if_yparxei, prob_spam_if_yparxei)
        if((synolo - (count_ham_yparxei + count_spam_yparxei)) == 0):
            prob_ham_if_d_yparxei = 0.5
        else:
            prob_ham_if_d_yparxei = (count_ham -count_ham_yparxei) / (synolo - (count_ham_yparxei + count_spam_yparxei))
        prob_spam_if_d_yparxei = 1 - prob_ham_if_d_yparxei
        h_if_d_yparxei = entropy(prob_ham_if_d_yparxei,prob_spam_if_d_yparxei)
        ig = (H - (prob_yparxei * h_if_yparxei) - (prob_d_yparxei * h_if_d_yparxei))
        if (ig > max_ig):
            max_ig = ig
            best_attr = a
    return best_attr

    
def accuracy(wrong, total):
	return (1 - (wrong / total)) * 100.0

def main(name):
    ham_emails = []
    spam_emails = []
    
    script_dir = os.path.dirname(__file__)
    
    rel_path_ham = name + "/ham"
    rel_path_spam =  name + "/spam"
    
    ham_path = os.path.join(script_dir, rel_path_ham)
    spam_path = os.path.join(script_dir, rel_path_spam)
    
    for file in os.listdir(ham_path):
        filepath = os.path.join(ham_path, file)
        with open(filepath, errors='ignore') as f:
            lines = f.read()
            ham_emails.append(lines)
    
    for file in os.listdir(spam_path):
        filepath = os.path.join(spam_path, file)
        with open(filepath, errors='ignore') as f:
            lines = f.read()
            spam_emails.append(lines)
    
    #split ham into train (80%), test(20%) 
    ham_len = len(ham_emails)
    ham_train_size = int(ham_len*0.8)
    ham_test_size = int(ham_len - ham_train_size)
    
    split_ham = [ham_train_size, ham_test_size]
    ham_emails_lists = [list(islice(ham_emails,elem)) for elem in split_ham]
    
    ham_train = ham_emails_lists[0]
    ham_test= ham_emails_lists[1]
    
    #split spam into train (80%), test(20%) 
    spam_len = len(spam_emails)
    spam_train_size = int(spam_len*0.8)
    spam_test_size = int(spam_len - spam_train_size)

    split_spam = [spam_train_size, spam_test_size]
    spam_emails_lists = [list(islice(spam_emails,elem)) for elem in split_spam]
    
    spam_train = spam_emails_lists[0]
    spam_test= spam_emails_lists[1]

    #ham C = 0
    #probability a message to be ham based on previous messages (train)
    ham_prop = ham_train_size/(ham_train_size + spam_train_size)
    #print (ham_prop)
    
    # spam C = 1
    #probability a message to be spam based on previous messages
    spam_prop = spam_train_size/(ham_train_size + spam_train_size)
    #print (spam_prop)
    
    #entropy -> how uncertain we are
    #h - entropy? H(c) = -P(C = 1) * logP(C = 1) - P(C = 0) * logP(C = 0)
    H = -1 * spam_prop * math.log(spam_prop, 2) - ham_prop * math.log(ham_prop, 2) 
    
    stop_words = set(stopwords.words('english'))
    stop_words.update(['Subject', ':',',','(',')','/','\'','-','\'\'','`', '``','?'])
    
    a = 0
    words = []
    in_ham_count = []
    in_spam_count = []
    
    ham_size = len(ham_train)
    spam_size = len(spam_train)
    
    #count words in ham emails
    for ham in ham_train:
        unique_ham = []
        for word in ham.split():
            if word not in (unique_ham and stop_words):
                unique_ham.append(word)
                if word in words:
                    in_ham_count[words.index(word)] += 1 
                else:
                   words.append(word)
                   in_ham_count.append(1)
                   in_spam_count.append(0)
    
    #count words in spam emails
    for spam in spam_train:
        unique_spam = []
        for x in spam.split():
            if x not in (unique_spam and stop_words):
                unique_spam.append(word)
                if word in words:
                    in_spam_count[words.index(word)] += 1 
                else:
                   words.append(word)
                   in_ham_count.append(0)
                   in_spam_count.append(1)
    
    ig_words = []
    
    # entropy(h) and information gain
    for a, word in enumerate(words):
        ham_and_spam = in_ham_count[a] + in_spam_count[a]
        prob_yparxei = (ham_and_spam / (ham_size + spam_size))
        prob_d_yparxei = 1 - prob_yparxei
        prob_ham_if_yparxei = (in_ham_count[a] / ham_and_spam) 
        prob_spam_if_yparxei = 1 - prob_ham_if_yparxei
        h_if_yparxei = entropy(prob_ham_if_yparxei, prob_spam_if_yparxei)
        prob_ham_if_d_yparxei = ((ham_size -in_ham_count[a]) / ((ham_size + spam_size) - ham_and_spam))
        prob_spam_if_d_yparxei = 1 - prob_ham_if_d_yparxei
        h_if_d_yparxei = entropy(prob_ham_if_d_yparxei,prob_spam_if_d_yparxei)
        ig = (H - (prob_yparxei * h_if_yparxei) - (prob_d_yparxei * h_if_d_yparxei))
        ig_words.append([ig, word])
        
    
    #sort in descending order
    ig_words.sort(reverse = True)
    top_ig = ig_words[:1000]
    
    sks = []
    for i in top_ig:
        sks.append(i[1])
    
    instance = create_instances(ham_train, spam_train, sks)
    
    c0 = []
    c1 = []
    
    for index in range(1000):
        c0.append(prob(instance, index + 1, 0))
        c1.append(prob(instance, index + 1, 1))
    
    
    results_train_nb = test_naive_bayes(ham_train, spam_train, c0, c1, sks, ham_prop, spam_prop)    
    results_test_nb = test_naive_bayes(ham_test, spam_test, c0, c1, sks, ham_prop, spam_prop)
    trained_tree = id3(instance, sks, 0)
    results_train_id3 = test_id3(trained_tree, ham_train, spam_train, words)
    results_test_id3 = test_id3(trained_tree, ham_test, spam_test, words)
    return [results_train_nb, results_test_nb, results_train_id3, results_test_id3 ]


results = []

for i in range (3):
    print("enron" + str(i + 1))
    results.append(main("enron" + str(i + 1)))
    
print (results)