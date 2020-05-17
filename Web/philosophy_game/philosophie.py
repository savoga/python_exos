#!/usr/bin/python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, session, request, redirect, flash, url_for
from getpage import getPage

app = Flask(__name__)

app.secret_key = "blablabla"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/new-game', methods=['POST'])
def newgame() :
    session['article'] = request.form['start']
    session['score'] = 0
    return redirect('/game')

@app.route('/game', methods=['GET'])
def game() :
    # app.logger.info(type(session['article']))
    session['title'], session['hrefs'] = getPage(session['article'])
    if session['title'] == 'Philosophie' :
        flash("Gagné ! Votre score est {}".format(session['score']), "victoire")
        return redirect('/')

    if session['title'] == [] :
        flash("Il n'y pas de lien associé !", "erreur")
        return redirect('/')

    if session['hrefs'] == None :
        flash("La page n'existe pas !", "erreur")
        return redirect('/')

    session['score'] += 1
    return render_template('game.html', title=session['title'], hrefs=session['hrefs'])

@app.route('/move', methods=['POST'])
def move() :
    # Si le joueur joue sur plusieurs onglets, on peut détecter
    # ce comportement si le lien sélectionné ne fait pas partie de la
    # liste des liens mémorisés dans la session
    if request.form['destination'] in session['hrefs'] :
        session['article'] = request.form['destination']
        return redirect('/game')
    else :
        flash("Vous jouez sur plusieurs onglets !", "erreur")
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)