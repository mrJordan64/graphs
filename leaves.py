import pygame
import numpy as np

#   WINDOW SPECS
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
HALF_WIDTH = int(WINDOW_WIDTH / 2)
HALF_HEIGHT = int(WINDOW_HEIGHT / 2)
FPS = 30

#   COLORS
WHITE  = [255, 255, 255]
BLACK  = [0,   0,   0]
GRAY   = [150, 150, 150]
RED    = [255, 0,   0]
GREEN  = [0,   255, 0]
BLUE   = [0,   0,   255]
YELLOW = [255, 255, 0]

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('3D GRAPHS')
clock = pygame.time.Clock()


def draw(leaf, color = BLACK):
    pygame.draw.circle(screen, BLACK, leaf[0], leaf[1]+2)
    pygame.draw.circle(screen, color, leaf[0], leaf[1])


n = 100
angle = 2*np.pi/1.618
d = 10

leaves = [[[0, 0], n-i] for i in range(n)]

game_over = False
while not game_over:
    clock.tick(FPS)
    screen.fill(WHITE)
    keys = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:
                x = 0

    for i in range(n):
        leaves[i][0] = [HALF_WIDTH+d*(n-i)*np.cos(i*angle), HALF_HEIGHT-d*(n-i)*np.sin(i*angle)]
        draw(leaves[i], GREEN)


    d += (keys[pygame.K_m] - keys[pygame.K_n])*0.25

    #put_text("FPS: " + str(round(clock.get_fps(), 5)), 16, BLACK, (8, 8))
    # put_text(str(camera.pos), 16, BLACK, (8, 20))

    pygame.display.update()

pygame.quit()
quit()
