import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygame

def draw_strokes_to_image(strokes, target_resolution, original_resolution):
    # Initialize a black image with the original resolution
    image = np.zeros([original_resolution[0], original_resolution[1]], dtype=np.uint8)

    # Draw each stroke
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for stroke in strokes:
        # Convert stroke points to array
        points = np.array(stroke, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Update min and max x, y values for cropping
        min_x, min_y = min(min_x, points[..., 0].min()), min(min_y, points[..., 1].min())
        max_x, max_y = max(max_x, points[..., 0].max()), max(max_y, points[..., 1].max())

        # Draw stroke on the image
        cv2.polylines(image, [points], isClosed=False, color=(255), thickness=16)

    # Determine cropping box
    width, height = max_x - min_x, max_y - min_y
    margin = 10
    if width > height:
        min_y = min_y - (width - height) // 2 - margin
        max_y = max_y + (width - height) // 2 + margin
    else:
        min_x = min_x - (height - width) // 2 - margin
        max_x = max_x + (height - width) // 2 + margin
    

    # Crop the image
    image = image[max(0, min_y):min(max_y, original_resolution[0]), max(0, min_x):min(max_x, original_resolution[1])]

    # Resize the image to target resolution
    image = cv2.resize(image, dsize=target_resolution, interpolation=cv2.INTER_CUBIC)

    image = image.astype(np.float32)/255.0

    return image


class GameDrawer:
    def __init__(self, screen, background, foreground) -> None:
        self.screen = screen
        self.background = background
        self.foreground = foreground

    def draw_current_stroke(self, current_stroke):
        if len(current_stroke) < 2:
            return
        pygame.draw.lines(self.screen, self.foreground, False, current_stroke, 5)
        pygame.display.flip()

    def __reset_screen(self):
        self.screen.fill(self.background)

    def __draw_character(self, character, position):
        self.screen.blit(character, position)

    def draw_initial_state(self, character, position):
        self.__reset_screen()
        self.__draw_character(character, position)
        pygame.display.flip()
    
    def draw_current_state(self, strokes, current_stroke, character, position):
        self.__reset_screen()
        self.__draw_character(character, position)

        # Draw previous strokes
        if len(strokes) != 0:
            for stroke in strokes:
                pygame.draw.lines(self.screen, self.foreground, False, stroke, 5)

        # Draw current stroke        
        if len(current_stroke) > 1:
            pygame.draw.lines(self.screen, self.foreground, False, current_stroke, 5)

        pygame.display.flip()