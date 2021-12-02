import numpy as np
import pygame
from pygame.locals import *

WIDTH = 700
HEIGHT = 700
SCREEN_CENTER = (np.array([WIDTH, HEIGHT]) / 2).astype(int)

WHITE  = [255, 255, 255]
BLACK  = [0,     0,   0]
RED    = [255,   0,   0]
GREEN  = [0,   255,   0]
BLUE   = [0,     0, 255]
YELLOW = [255, 255,   0]


def put_text(text, position, size=16, color=BLACK):
    txt = pygame.font.SysFont("minecraftiaregular", size).render(text, 1, color)
    screen.blit(txt, position)


def screen_pos(position):
    return ([position[0], -position[1]] + SCREEN_CENTER).astype(int)


def draw_circle(position, color=BLACK, radius=2):
    pygame.draw.circle(screen, color, screen_pos(position), radius)


def draw_line(position1, position2, color=BLACK, thickness=1):
    pygame.draw.line(screen, color, screen_pos(position1), screen_pos(position2), thickness)


def draw_arc(point, radius, start_angle, end_angle, color=BLACK, thickness=1):
    _point = screen_pos(point)
    radius = int(radius)
    rect = pygame.Rect(_point[0]-radius, _point[1]-radius, 2 * radius, 2 * radius)
    pygame.draw.arc(screen, color, rect, start_angle, end_angle, thickness)


def angle_3pts(a, b, c):
    ba = a-b
    bc = c-b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))

    return np.arccos(cos_angle)
    # return np.degrees(np.arccos(cos_angle))


def angle_2pts(a, b):
    vect = b-a
    return np.arctan2(vect[1], vect[0])


def dist(position1, position2):
    return np.linalg.norm(np.array(position1)-np.array(position2))


class Polygon:
    def __init__(self, n, r):
        self.n = n
        self.r = r
        self.points = r*np.array([[np.cos(i*2*np.pi/n), np.sin(i*2*np.pi/n)] for i in range(n)])

    def draw(self):
        for i in range(self.n):
            draw_line(self.points[i], self.points[(i+1) % self.n])
            draw_circle(self.points[i])


# reflects point with respect to side1 onto side2, with option to reverse reflection
def reflect_point(point, polygon, side1, side2, reverse):
    # depending on reverse, calculate angle and distance from point to either extremes of the side
    if reverse:
        angle_pt_side1 = angle_3pts(polygon.points[(side1+1) % polygon.n], polygon.points[side1], point)
        dist_pt_ptside1 = dist(polygon.points[side1], point)
    else:
        angle_pt_side1 = angle_3pts(polygon.points[side1], polygon.points[(side1+1) % polygon.n], point)
        dist_pt_ptside1 = dist(polygon.points[(side1+1) % polygon.n], point)

    angle_side2 = angle_2pts(polygon.points[side2], polygon.points[(side2+1) % polygon.n])
    new_angle = angle_side2-angle_pt_side1

    return polygon.points[side2]+dist_pt_ptside1*np.array([np.cos(new_angle), np.sin(new_angle)])


def pt_in_sight(pt1, pt2, side, polygon):
    angle = angle_3pts(polygon.points[side], pt1, polygon.points[(side+1) % polygon.n])
    if angle_3pts(polygon.points[(side+1) % polygon.n], pt1, pt2) < angle and\
            angle_3pts(polygon.points[side], pt1, pt2) < angle:
        return True
    else:
        return False


# RETURNS TRUE IF P1 IS LEFT AND P2 RIGHT FROM P'S PERSPECTIVE, ELSE FALSE
# WHICH IS WHEN P CAN SEE SEGMENT P1P2 OF THE POLYGON FROM THE 'INSIDE'
def p1L_p2R(p, p1, p2):
    angle1 = angle_2pts(p, p1)
    if angle1 < 0:
        angle1 = 2 * np.pi + angle1
    angle2 = angle_2pts(p, p2)
    if angle2 < 0:
        angle2 = 2 * np.pi + angle2

    if angle1 < angle2 and angle2-angle1 > np.pi:
        return True
    elif angle1 > angle2 and angle1-angle2 < np.pi:
        return True
    else:
        return False


'''
ALGORYTHM:

def min_dist(p1, p2, poly):
    distance = dist(p1, p2)
    for '''


radius = 200
sides = 8
polygon = Polygon(sides, radius)

point = np.array([100, 150])
point2 = np.array([-80, -150])

side2 = 6

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poly")
clock = pygame.time.Clock()
end = False
while not end:
    time = clock.tick(30)
    screen.fill(WHITE)
    for event in pygame.event.get():
        keys = pygame.key.get_pressed()
        if event.type == QUIT:
            end = True

    point[0] += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
    point[1] += (keys[pygame.K_UP] - keys[pygame.K_DOWN])
    side2 = (side2 + keys[pygame.K_SPACE]) % polygon.n

    polygon.draw()
    draw_circle(point)
    draw_circle(point2)
    d1 = dist(point, point2)
    reflected_pt = reflect_point(point, polygon, 1, side2, True)
    d2 = dist(reflected_pt, point2)
    draw_circle(reflected_pt)

    rect = pygame.Rect(350 - 50, 350 - 20, 2 * 50, 2 * 20)
    pygame.draw.arc(screen, BLACK, rect, 0, 2*np.pi, 2)



    if pt_in_sight(reflected_pt, point2, side2, polygon):
        in_sight = True
        draw_circle(point2, RED, 3)
    else:
        in_sight = False

    if d2 < d1 and in_sight:
        distance = d2
        draw_line(reflected_pt, point2)
    else:
        distance = d1
        draw_line(point, point2)

    for i in range(polygon.n):
        p1 = polygon.points[(i+1) % polygon.n]
        p2 = polygon.points[i]

        if p1L_p2R(reflected_pt, p1, p2):
            draw_line(p1, p2, RED)

    '''
    start_angle = angle_2pts(reflected_pt, polygon.points[(side2+1) % polygon.n])
    end_angle = angle_2pts(reflected_pt, polygon.points[side2])
    #draw_line(reflected_pt, reflected_pt+distance*)
    draw_line(reflected_pt, polygon.points[(side2 + 1) % polygon.n])
    draw_arc(reflected_pt, distance, start_angle, end_angle)'''

    # put_text("DIST: " + str(round(dist(point, point2), 3)), (8, 8))
    put_text("ANGLE: " + str(round(angle_2pts(point2, point), 3)), (8, 8))

    pygame.display.update()

pygame.quit()