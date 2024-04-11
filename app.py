import pygame
import sys
from characters import get_hiragana_characters
import random
from models import TMClassifier
import csv

pygame.init()

width, height = 800, 800
screen = pygame.display.set_mode((width, height))

black = (0, 0, 0)
white = (255, 255, 255)
screen.fill(white)

strokes = []
current_stroke = []

# Loading the character data
with open('k49_classmap.csv') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    class_translation_table = {int(label): desc for label, _, desc in reader}

characters = get_hiragana_characters() 
current_character = random.choice(characters)
text_to_display = []

# Display first character to draw
font = pygame.font.Font(None, 36)
hiragana_text = font.render(current_character.romanji, True, black)
text_to_display.append((hiragana_text, (width - 80, 50)))
screen.blit(text_to_display[-1][0], text_to_display[-1][1])

pygame.display.flip()

tm = TMClassifier("/home/olepedersen/source/repos/hiragana_practice_tool/model.pk1", class_translation_table)

nr_drawn_points = 0

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Start stroke 
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            current_stroke = [event.pos]

        # Draw stroke 
        if event.type == pygame.MOUSEMOTION and nr_drawn_points > 0:
            current_stroke.append(event.pos)

        # Finish stroke 
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            strokes.append(current_stroke)
            current_stroke = []
            nr_drawn_points = 0
            if len(strokes) == current_character.nr_of_strokes:
                prediction = tm.predict(strokes)
                if prediction == current_character.character:
                    print("Correct!")
                else: 
                    print("Wrong!")
                strokes = []
                current_character = random.choice(characters)

        # Undo last stroke
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if len(current_stroke) > 0:
                current_stroke = []
                num_points = 0
            if len(current_stroke) > 0:
                current_stroke = []
            elif len(strokes) > 0:
                strokes.pop()
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit()
        
        # Update screen
        if len(current_stroke) > nr_drawn_points:
            screen.fill(white)
            if len(strokes) > 0:
                for stroke in strokes:
                    pygame.draw.lines(screen, black, False, stroke, 5)
            if len(current_stroke) > 1:
                pygame.draw.lines(screen, black, False, current_stroke, 5)
            for text, position in text_to_display:
                screen.blit(text, position)
            pygame.display.flip()
            nr_drawn_points += 1