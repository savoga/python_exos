# Written with <3 by Julien Romero

import hashlib
from sys import argv
import sys
import numpy as np
if (sys.version_info > (3, 0)):
    from urllib.request import urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import urlopen
    from urllib import urlencode
import itertools
import random as random


NAME = "savoure".lower()
# This is the correct location on the moodle
ENCFILE = "../passwords2020/" + NAME + ".enc"
# If you run the script on your computer: uncomment and fill the following
# line. Do not forget to comment this line again when you submit your code
# on the moodle.
ENCFILE = "savoure.enc"



class Crack:
    """Crack The general method used to crack the passwords"""

    def __init__(self, filename, name):
        """__init__
        Initialize the cracking session
        :param filename: The file with the encrypted passwords
        :param name: Your name
        :return: Nothing
        """
        self.name = name.lower()
        self.passwords = get_passwords(filename)

    def check_password(self, password):
        """check_password
        Checks if the password is correct
        !! This method should not be modified !!
        :param password: A string representing the password
        :return: Whether the password is correct or not
        """
        password = str(password)
        cond = False
        if (sys.version_info > (3, 0)):
            cond = hashlib.md5(bytes(password, "utf-8")).hexdigest() in \
                self.passwords
        else:
            cond = hashlib.md5(bytearray(password)).hexdigest() in \
                self.passwords
        if cond:
            args = {"name": self.name,
                    "password": password}
            args = urlencode(args, "utf-8")
            page = urlopen('http://137.194.211.71:5000/' +
                                          'submit?' + args)
            if b'True' in page.read():
                print("You found the password: " + password)
                return True
        return False

    def crack(self):
        """crack
        Cracks the passwords. YOUR CODE GOES BELOW.

        We suggest you use one function per question. Once a password is found,
        it is memorized by the server, thus you can comment the call to the
        corresponding function once you find all the corresponding passwords.
        """
        self.bruteforce_digits()
        self.bruteforce_letters()

        self.dictionary_passwords()
        self.dictionary_passwords_leet()
        self.dictionary_words_hyphen()
        self.dictionary_words_digits()
        self.dictionary_words_diacritics()
        self.dictionary_city_diceware()

        self.social_google()
        self.social_jdoe()
        self.paste()

    def bruteforce_digits(self):
        num = ['0','1','2','3','4','5','6','7','8','9']
        for r in range(1,10) :
            for s in itertools.product(num, repeat=r):
                self.check_password(''.join(s))
        pass

    def bruteforce_letters(self):
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        for r in range(1,6) :
            for s in itertools.product(letters, repeat=r):
                self.check_password(''.join(s))
        pass

    def dictionary_passwords(self):
        a = open('10k_most_common.txt','r')
        b = a.readlines()
        for l in b:
            self.check_password(l.replace('\n',''))
        pass

    def dictionary_passwords_leet(self):
        a = open('10k_most_common.txt','r')
        mostCommonWords = a.readlines()
        for word in mostCommonWords:
            word = word.replace('\n','')
            replacements = (('a','4'), ('e','3'),('l','1'), ('o','0'), ('t','+'),
                            ('a','@'), ('i','1'))
            new_word = word
            for old, new in replacements:
                new_word = new_word.replace(old, new)
                self.check_password(new_word)
        pass

    def dictionary_words_hyphen(self):
        a = open('20k_most_common_eng.txt','r')
        mostCommonWords = a.readlines()
        for word in mostCommonWords:
            word = word.replace('\n','')
            for i in range(len(word)):
                word_1 = ''.join([word[:i],'-',word[i:]])
                self.check_password(word_1)
                for j in range(len(word_1)):
                    word_2 = ''.join([word_1[:j],'-',word_1[j:]])
                    self.check_password(word_2)
                    for k in range(len(word_2)):
                        word_3 = ''.join([word_2[:k],'-',word_2[k:]])
                        self.check_password(word_3)
        pass

    def dictionary_words_digits(self):
        num = ['0','1','2','3','4','5','6','7','8','9']
        a = open('20k_most_common_eng.txt','r')
        mostCommonWords = a.readlines()
        for word1 in mostCommonWords:
            word1 = word1.replace('\n','')
            for word2 in mostCommonWords:
                word2 = word2.replace('\n','')
                if(word2 == word1):
                    continue
                w = word1 + word2
                for s in itertools.product(num, repeat=2):
                    finalWord = ''.join(str(w) + str(''.join(s)))
                    self.check_password(finalWord)
        pass

    def dictionary_words_diacritics(self):
        a = open('10k-most-common_fr.txt','r')
        mostCommonWords = a.readlines()

        for word in mostCommonWords:
            word = word.split("\t")[0]
            replacements = (('é','e'), ('è','e'),('à','a'),('ç','c'),('ù','u'),
                            ('ô','o'),('â','a'),('î','i'),('û','u'),('ê','e'))
            for old, new in replacements:
                new_word = word.replace(old, new, 1)
                self.check_password(new_word)
        pass

    def dictionary_city_diceware(self):
        pass

    def social_google(self):
        pass

    def social_jdoe(self):
        pass

    def paste(self):
        pass


def get_passwords(filename):
    """get_passwords
    Get the passwords from a file
    :param filename: The name of the file which stores the passwords
    :return: The set of passwords
    """
    passwords = set()
    with open(filename, "r") as f:
        for line in f:
            passwords.add(line.strip())
    return passwords


# First argument is the password file, the second your name
crack = Crack(ENCFILE, NAME)


if __name__ == "__main__":
    crack.crack()