import pygame
import sys
from characters import get_hiragana_characters
import random
from models import TMClassifier
from config import WIDTH, HEIGHT, FOREGROUND, BACKGROUND, X_HIRAGANA, Y_HIRAGANA, TIMEOUT, CHECK_MARK_IMG_PATH, X_MARK_IMG_PATH
from utils import GameDrawer, get_class_translation_table
from functools import partial

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))

strokes = []
current_stroke = []
characters = get_hiragana_characters() 
current_character = random.choice(characters)

# Display first character to draw
romanji_font = pygame.font.Font(None, 36)
hiragana_font = pygame.font.Font("static/NotoSansJP-Black.ttf", 36)
hiragana_text = romanji_font.render(current_character.romanji, True, FOREGROUND)

game_drawer = GameDrawer(screen, BACKGROUND, FOREGROUND)
game_drawer.draw_initial_state(hiragana_text, (X_HIRAGANA, Y_HIRAGANA))

tm = TMClassifier("/home/olepedersen/source/repos/hiragana_practice_tool/model.pk1", get_class_translation_table())

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
                set_current_state = partial(game_drawer.draw_current_state, strokes, current_stroke, hiragana_font.render(current_character.character, True, FOREGROUND), (X_HIRAGANA, Y_HIRAGANA))
                if prediction == current_character.character:
                    print("Correct!")
                    game_drawer.draw_mark(set_current_state, CHECK_MARK_IMG_PATH, TIMEOUT // 1000)
                else:
                    print("Wrong!")
                    game_drawer.draw_mark(set_current_state, X_MARK_IMG_PATH, TIMEOUT // 1000)
                strokes = []
                current_character = random.choice(characters)
                hiragana_text = romanji_font.render(current_character.romanji, True, FOREGROUND)
                game_drawer.draw_current_state(strokes, current_stroke, hiragana_text, (X_HIRAGANA, Y_HIRAGANA))

        # Undo last stroke
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if len(current_stroke) > 0:
                current_stroke = []
                game_drawer.draw_current_state(strokes, current_stroke, hiragana_text, (X_HIRAGANA, Y_HIRAGANA))
            elif len(strokes) > 0:
                strokes.pop()
                game_drawer.draw_current_state(strokes, current_stroke, hiragana_text, (X_HIRAGANA, Y_HIRAGANA))
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit()