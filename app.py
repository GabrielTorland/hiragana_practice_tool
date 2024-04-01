import pygame
import sys
from characters import get_hiragana_characters

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))

black = (0, 0, 0)
white = (255, 255, 255)
screen.fill(white)
pygame.display.flip()

strokes = []
current_stroke = []
characters = get_hiragana_characters()

def draw_strokes():
    screen.fill(white)  # Clear screen
    for stroke in strokes:
        if len(stroke) > 1:
            pygame.draw.lines(screen, black, False, stroke, 5)
    pygame.display.flip()

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
            draw_strokes()           
            current_stroke = []
            nr_drawn_points = 0
            
        # Undo last stroke
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if len(current_stroke) > 0:
                current_stroke = []
                draw_strokes()
                num_points = 0
            if len(current_stroke) > 0:
                current_stroke = []
                draw_strokes()
            elif len(strokes) > 0:
                strokes.pop()
                draw_strokes()
        
        # Update screen
        if len(current_stroke) > nr_drawn_points:
            screen.fill(white)
            if len(strokes) > 0:
                for stroke in strokes:
                    pygame.draw.lines(screen, black, False, stroke, 5)
            if len(current_stroke) > 1:
                pygame.draw.lines(screen, black, False, current_stroke, 5)
            pygame.display.flip()
            nr_drawn_points += 1