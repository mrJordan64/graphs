import numpy as np
import pygame
from pygame.locals import *

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
SCREEN_CENTER = (np.array([WINDOW_WIDTH, WINDOW_HEIGHT]) / 2).astype(int)

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
YELLOW = [255, 255, 0]

np.random.seed(42)


def put_text(text, position, size=16, color=BLACK):
    txt = pygame.font.SysFont("minecraftiaregular", size).render(text, 1, color)
    screen.blit(txt, position)


def screen_pos(position):
    return ([position[0], -position[1]] + SCREEN_CENTER).astype(int)


def draw_circle(center, radius=2, color=BLACK):
    pygame.draw.circle(screen, color, center.screenPos(), radius, 1)


def draw_line(position1, position2, color=BLACK, thickness=1):
    pygame.draw.line(screen, color, screen_pos(position1), screen_pos(position2), thickness)


def draw_arc(point, radius, start_angle, end_angle, color=BLACK, thickness=1):
    pos = screen_pos(point)
    rect = pygame.Rect(pos[0] - int(radius), pos[1] - int(radius), 2 * int(radius), 2 * int(radius))
    pygame.draw.arc(screen, color, rect, start_angle, end_angle, thickness)


def angle_3pts(A, B, C):  # returns angle between B (middle) and A, C
    BA = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))

    return np.arccos(cos_angle)
    # return np.degrees(np.arccos(cos_angle))


def angle_2pts(A, B):
    vectAB = [B.x - A.x, B.y - A.y]
    return np.arctan2(vectAB[1], vectAB[0])


def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def get_polygon(sides, radius):
    angle = 2 * np.pi / sides
    return [Point([radius*np.cos(i*angle), radius*np.sin(i*angle)]) for i in range(sides)]

def draw_polygon(polygon):
    for i in range(len(polygon)):
        polygon[i].connect_to(polygon[(i + 1) % len(polygon)])
        polygon[i].draw()

class Polygon:
    def __init__(self, sides, radius):
        self.sides = sides
        self.radius = radius
        angle = 2*np.pi/sides
        self.points = [Point([radius*np.cos(i*angle), radius*np.sin(i*angle)]) for i in range(sides)]

    def draw(self):
        for i in range(self.sides):
            self.points[i].connect_to(self.points[(i + 1) % self.sides])
            self.points[i].draw()


class Point:
    def __init__(self, position, color=BLACK):
        self.x = position[0]
        self.y = position[1]
        self.color = color

    def __add__(self, point):
        return Point([self.x+point.x, self.y+point.y])

    def __sub__(self, point):
        return Point([self.x-point.x, self.y-point.y])

    def screen_pos(self):
        return (SCREEN_CENTER + [self.x, -self.y]).astype(int)

    def draw(self, color=BLACK, size=2):
        pygame.draw.circle(screen, color, self.screen_pos(), size)

    def distance(self, point):
        return np.linalg.norm([self.x-point.x, self.y-point.y])

    def connect_to(self, point, color=BLACK, thickness=1):
        pygame.draw.line(screen, color, self.screen_pos(), point.screenPos(), thickness)

    # reflects point with respect to side1 onto side2 depending on reversed reflection
    def reflected(self, polygon, side1, side2, reversed):
        # angle from polygon.points[(side1+1) % polygon.sides] to point with respect to polygon.points[side1]
        angle1 = angle_3pts(polygon[(side1+1) % len(polygon)], polygon[side1], point)
        dist = self.distance(polygon[side1])

        if not reversed:
            # angle of side2 from polygon.points[side2] to polygon.points[(side2+1) % polygon.sides]
            angle2 = angle_2pts(polygon[side2], polygon[(side2 + 1) % len(polygon)])
            newAngle = angle2 - angle1
            # with center at polygon.points[side2], put new pt at radius dist and new angle
            return polygon[side2] + Point(dist * np.array([np.cos(newAngle), np.sin(newAngle)]))

        else:
            angle2 = angle_2pts(polygon[(side2+1) % len(polygon)], polygon[side2])
            newAngle = angle2 + angle1
            return polygon[(side2+1) % len(polygon)] + Point(dist * np.array([np.cos(newAngle), np.sin(newAngle)]))


def pt_in_sight(pt1, pt2, side, polygon):  # pt1 inside polygon!
    angle = angle_3pts(polygon[side], pt1, polygon[(side + 1) % len(polygon)])
    if angle_3pts(polygon[(side + 1) % len(polygon)], pt1, pt2) < angle and \
            angle_3pts(polygon[side], pt1, pt2) < angle:
        return True
    else:
        return False


# RETURNS TRUE IF P1 IS LEFT AND P2 RIGHT FROM P'S PERSPECTIVE, ELSE FALSE
# WHICH IS WHEN P CAN SEE SEGMENT P1P2 OF THE POLYGON FROM THE 'INSIDE'
def p1L_p2R(p, p1, p2): ######################################################################################### WORKS BUT WTF
    angle1 = angle_2pts(p, p1)
    if angle1 < 0:
        angle1 = 2 * np.pi + angle1
    angle2 = angle_2pts(p, p2)
    if angle2 < 0:
        angle2 = 2 * np.pi + angle2

    if angle1 < angle2 and angle2 - angle1 > np.pi:
        return True
    elif angle1 > angle2 and angle1 - angle2 < np.pi:
        return True
    else:
        return False


def random_word(length): # generate list of  ordered randomly
    word = [i for i in range(-int(length/2), int(length/2)+1) if i!=0]
    for i in range(length):
        word[i] *= np.random.choice([-1,1])
    np.random.shuffle(word)
    return word


def reflection_edge(word, i):
    for j in range(len(word)):
        if j!=i and abs(word[j]) == abs(word[i]):
            return j
            break


def draw_arrow(p1, p2):
    pt = p1 + Point([0.9*(p2 - p1).x, 0.9*(p2 - p1).y])
    pt.draw(color=BLUE, size=5)


sides = 8
radius = 200
word = random_word(sides)
reflectionEdges = [reflection_edge(word, i) for i in range(sides)]
reversions = [word[i]*word[reflectionEdges[i]] < 0 for i in range(sides)]

print(word)
print(reflectionEdges)

polygon = get_polygon(sides, radius)
point = Point(np.array([100, 150]))
point2 = Point(np.array([-80, -150]))

minDist = 0

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
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

    point.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
    point.y += (keys[pygame.K_UP] - keys[pygame.K_DOWN])
    point2.x += (keys[pygame.K_d] - keys[pygame.K_a])
    point2.y += (keys[pygame.K_w] - keys[pygame.K_s])

    draw_polygon(polygon)
    point.draw(color=RED, size=5)
    point2.draw(color=RED, size=5)

    distNormal = point2.distance(point)
    minDist = distNormal
    draw_circle(point2, distNormal)

    # draw blue circles to mark direction of polygon sides
    for i in range(len(polygon)):
        middlePt = polygon[i]+polygon[(i+1)%len(polygon)]
        middlePt.x, middlePt.y = middlePt.x/2, middlePt.y/2

        pos = (middlePt.screen_pos()).astype(int)
        put_text(str(word[i]), (pos[0], pos[1]))
        if word[i] > 0:
            draw_arrow(polygon[i], polygon[(i+1)%len(polygon)])
        else:
            draw_arrow(polygon[(i + 1) % len(polygon)], polygon[i])

    reflectionPts = [] # list of [reflectedPoint, reflectedEdge]
    for edge in range(2):
        reflectionPoint = point.reflected(polygon, edge, reflectionEdges[edge], reversions[edge])
        reflectionPoint.draw()
        reflectionPts.append([reflectionPoint, reflectionEdges[edge]])

    reflectionPts.sort(key=lambda reflection: point2.distance(reflection[0]))

    for i in range(2):
        if pt_in_sight(point2, reflectionPts[i][0], reflectionPts[i][1], polygon) \
                and point2.distance(reflectionPts[i][0]) < distNormal:
            point2.connect_to(reflectionPts[i][0])
            minDist = point2.distance(reflectionPts[i][0])
            break
    else:
        point2.connect_to(point)

    for i in range(2):
        angle1 = angle_2pts(reflectionPts[i][0], polygon[(reflectionPts[i][1] + 1) % len(polygon)])
        angle2 = angle_2pts(reflectionPts[i][0], polygon[reflectionPts[i][1]])
        pointAngle1 = reflectionPts[i][0] + Point(minDist*np.array([np.cos(angle1), np.sin(angle1)]))
        pointAngle2 = reflectionPts[i][0] + Point(minDist*np.array([np.cos(angle2), np.sin(angle2)]))
        reflectionPts[i][0].connect_to(pointAngle1)
        reflectionPts[i][0].connect_to(pointAngle2)
        # rectangle_pt = Point([reflectionPts[i][0].x - minDist, reflectionPts[i][0].y + minDist]).screen_pos()
        # rectangle = pygame.Rect(rectangle_pt[0], rectangle_pt[1], 2*minDist, 2*minDist)
        #pygame.draw.arc(screen, BLACK, rectangle, angle1, angle2, 2)

    '''
    for i in range(len(polygon)):
        p1 = polygon[(i + 1) % len(polygon)]
        p2 = polygon[i]

        if not p1L_p2R(reflectedPoint, p1, p2):
            p1.connect_to(p2, RED)
    '''

    '''
    start_angle = angle_2pts(reflectedPoint, polygon.points[(reflectionSide+1) % polygon.n])
    end_angle = angle_2pts(reflectedPoint, polygon.points[reflectionSide])
    #draw_line(reflectedPoint, reflectedPoint+distance*)
    draw_line(reflectedPoint, polygon.points[(reflectionSide + 1) % polygon.n])
    draw_arc(reflectedPoint, distance, start_angle, end_angle)'''

    # put_text("DIST: " + str(round(dist(point, point2), 3)), (8, 8))
    put_text("ANGLE: " + str(round(angle_2pts(point2, point), 3)), (8, 8))

    pygame.display.update()

pygame.quit()
