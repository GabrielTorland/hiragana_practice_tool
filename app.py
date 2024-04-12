import pygame
import sys
from characters import get_hiragana_characters
import random
from models import TMClassifier
from utils import GameDrawer
import csv

pygame.init()

width, height = 800, 800
screen = pygame.display.set_mode((width, height))

foreground = (0, 0, 0)
background = (255, 255, 255)
screen.fill(background)

strokes = []
current_stroke = []

# Loading the character data
with open('k49_classmap.csv') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    class_translation_table = {int(label): desc for label, _, desc in reader}

characters = get_hiragana_characters() 
current_character = random.choice(characters)

# Display first character to draw
font = pygame.font.Font(None, 36)
hiragana_text = font.render(current_character.romanji, True, foreground)

game_drawer = GameDrawer(screen, background, foreground)
game_drawer.draw_initial_state(hiragana_text, (width - 80, 50))

tm = TMClassifier("/home/olepedersen/source/repos/hiragana_practice_tool/model.pk1", class_translation_table)

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Start stroke 
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            current_stroke = [event.pos]

        # Draw stroke 
        if event.type == pygame.MOUSEMOTION and len(current_stroke) > 0:
            current_stroke.append(event.pos)
            if len(current_stroke) % 10 == 0:
                game_drawer.draw_current_stroke(current_stroke)

        # Finish stroke 
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            strokes.append(current_stroke)
            current_stroke = []
            if len(strokes) == current_character.nr_of_strokes:
                prediction = tm.predict(strokes)
                if prediction == current_character.character:
                    print("Correct!")
                else: 
                    print("Wrong!")
                strokes = []
                current_character = random.choice(characters)
                hiragana_text = font.render(current_character.romanji, True, foreground)
                game_drawer.draw_current_state(strokes, current_stroke, hiragana_text, (width - 80, 50))

        # Undo last stroke
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if len(current_stroke) > 0:
                current_stroke = []
                game_drawer.draw_current_state(strokes, current_stroke, hiragana_text, (width - 80, 50))
            elif len(strokes) > 0:
                strokes.pop()
                game_drawer.draw_current_state(strokes, current_stroke, hiragana_text, (width - 80, 50))
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit()