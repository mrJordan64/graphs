'''
INSTRUCTIONS:

- o/p keys to decrease/increase edge elasticity coef
- k/l keys to decrease/increase node repulsion coef
- n/m keys to decrease/increase zoom
- left click & drag to move nodes
- left click an edge to add a node to it
- right click to delete edges
- right click to select a node and...
    ...right click again to unselect it
    ...press backspace to delete it (and all it's edges!)
    ...select another node to connect them
    ...click on empty space to create a new node connected to it
    ...press f to fix it to its place
'''

import numpy as np
import pygame
from pygame.locals import *

WIDTH = 640
HEIGHT = 480
SCREEN_CENTER = (np.array([WIDTH, HEIGHT]) / 2).astype(int)

WHITE  = [255, 255, 255]
BLACK  = [0,     0,   0]
RED    = [255,   0,   0]
GREEN  = [0,   255,   0]
BLUE   = [0,     0, 255]
YELLOW = [255, 255,   0]

friction_coef = 0.9
elastic_coef = 0.01
repel_coef = 5000

NONE = -1
center = np.array([0., 0.])
zoom = 1

node_size = 10
NODE_NUM = 10

'''
ADJ_MTX = np.zeros((NODE_NUM,NODE_NUM))
for i in range(NODE_NUM*2):
    i, j = np.random.randint(NODE_NUM), np.random.randint(NODE_NUM)
    ADJ_MTX[i,j] = 1
    ADJ_MTX[j,i] = 1

#ADJ_MTX = np.ones((NODE_NUM,NODE_NUM))'''

ADJ_MTX = np.array([[0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]])


class Node:
    def __init__(self, idx, pos):
        self.idx = idx
        self.pos = pos
        self.screen_pos = screen_pos(pos)
        self.speed = np.array([0., 0.])
        self.selected = False
        self.fixed = False

    def update_speed(self, nodes):
        if not self.fixed:
            for node in nodes:
                # ADD REPELLING FORCES FROM OTHER NODES
                if node_dist_mtx[self.idx, node.id] > node_size:
                    self.speed -= repel_coef * node_vect_mtx[self.idx, node.id] / (node_dist_mtx[self.idx, node.id] ** 3)
                    # DIST^3 BECAUSE WE ALSO NORMALIZE VECT

                # ADD ATTRACTIVE FORCES FROM EDGES
                if ADJ_MTX[self.idx, node.id]:
                    self.speed += node_vect_mtx[self.idx, node.id] * elastic_coef

        self.speed *= friction_coef

    def draw(self):
        if self.fixed:
            pygame.draw.circle(screen, BLUE, self.screen_pos, node_size)
        else:
            pygame.draw.circle(screen, BLACK, self.screen_pos, node_size)


def screen_pos(position):
    return ((position - center) * zoom + SCREEN_CENTER).astype(int)


def put_text(text, position, size=16, color=BLACK):
    txt = pygame.font.SysFont("minecraftiaregular", size).render(text, 1, color)
    screen.blit(txt, position)


def graph_center():
    center = np.array([0, 0])
    for node in nodes:
        center = center + node.pos

    return center / len(nodes)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Graph")
clock = pygame.time.Clock()

nodes = [Node(i, np.random.rand(2) * 100) for i in range(NODE_NUM)]
# nodes[0].pos = np.array([0.,0.])

click = False  # True if a mouse button has been clicked in that frame
highlighted = NONE  # index of highlighted node
selected = NONE  # index of selected node
dragged = NONE  # index of dragged node

node_dist_mtx = np.zeros((NODE_NUM, NODE_NUM))
node_vect_mtx = np.zeros((NODE_NUM, NODE_NUM, 2))

end = False
while not end:
    time = clock.tick(30)
    screen.fill(WHITE)
    click = False
    highlighted = NONE
    edge_edited = False
    center = graph_center()

    for event in pygame.event.get():
        keys = pygame.key.get_pressed()
        mouse_button = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        if event.type == QUIT:
            end = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        if event.type == pygame.MOUSEBUTTONUP and dragged != NONE:
            dragged = NONE

    elastic_coef += (keys[pygame.K_p] - keys[pygame.K_o]) * 0.001
    repel_coef += (keys[pygame.K_l] - keys[pygame.K_k]) * 100
    zoom += (keys[pygame.K_m] - keys[pygame.K_n]) * 0.05
    node_size = abs(int(10 * zoom))

    # CALCULATE VECTORS AND DISTANCES BETWEEN NODES
    for i in range(NODE_NUM):
        for j in range(i, NODE_NUM):
            node_vect_mtx[i, j] = nodes[j].pos - nodes[i].pos
            node_vect_mtx[j, i] = -1 * node_vect_mtx[i, j]
            node_dist_mtx[i, j] = np.linalg.norm(node_vect_mtx[i, j])
            node_dist_mtx[j, i] = node_dist_mtx[i, j]

    # FIX NODE
    if keys[pygame.K_f] and selected != NONE:
        nodes[selected].fixed = not nodes[selected].fixed
        selected = NONE

    # DRAG NODE
    if dragged != NONE:
        mouse_real_pos = (np.array(mouse_pos) - SCREEN_CENTER) / zoom + center
        nodes[dragged].speed += (mouse_real_pos - nodes[dragged].pos) * 0.1*friction_coef

    print(nodes[1].speed is nodes[2].speed)

    # UPDATE SPEEDS AND POSITIONS OF NODES
    for i in range(NODE_NUM):
        nodes[i].update_speed(nodes)
        nodes[i].pos += nodes[i].speed
        nodes[i].screen_pos = screen_pos(nodes[i].pos)

    # DRAW AND EDIT EDGES
    for i in range(NODE_NUM):
        for j in range(i, NODE_NUM):
            if ADJ_MTX[nodes[i].idx][nodes[j].idx]:  # (IF NODES ARE CONNECTED)
                d1 = np.linalg.norm(mouse_pos - nodes[i].screen_pos)
                d2 = np.linalg.norm(mouse_pos - nodes[j].screen_pos)
                d = np.linalg.norm(nodes[i].screen_pos - nodes[j].screen_pos)
                # IF MOUSE IN ELLIPSE NEAR EDGE, DRAW EDGE HIGHLIGHTED
                if d1 > node_size and d2 > node_size and d1 + d2 - d < 1.5:
                    pygame.draw.line(screen, RED, nodes[i].screen_pos, nodes[j].screen_pos, 5)
                    if click and not edge_edited:
                        edge_edited = True  # WE ONLY WANT TO EDIT MAX ONE EDGE PER CLICK
                        # DELETE EDGE
                        if mouse_button[2]:  # if right click
                            ADJ_MTX[nodes[i].idx, nodes[j].idx] = 0
                            ADJ_MTX[nodes[j].idx, nodes[i].idx] = 0

                        # ADD NODE IN MIDDLE OF EDGE
                        elif mouse_button[0]:  # if left click
                            ADJ_MTX[nodes[i].idx, nodes[j].idx] = 0
                            ADJ_MTX[nodes[j].idx, nodes[i].idx] = 0

                            pos = (nodes[i].pos + nodes[j].pos) / 2  # position of new node

                            nodes = np.append(nodes, Node(NODE_NUM, pos))
                            NODE_NUM += 1
                            ADJ_MTX = np.append(ADJ_MTX, np.zeros((1, NODE_NUM - 1)), axis=0)
                            ADJ_MTX = np.append(ADJ_MTX, [[0] for i in range(NODE_NUM)], axis=1)
                            ADJ_MTX[nodes[i].id, NODE_NUM - 1] = 1
                            ADJ_MTX[NODE_NUM - 1, nodes[i].id] = 1
                            ADJ_MTX[nodes[j].id, NODE_NUM - 1] = 1
                            ADJ_MTX[NODE_NUM - 1, nodes[j].id] = 1
                            node_dist_mtx = np.zeros((NODE_NUM, NODE_NUM))
                            node_vect_mtx = np.zeros((NODE_NUM, NODE_NUM, 2))

                # ELSE DRAW EDGE NORMALLY
                else:
                    pygame.draw.line(screen, BLACK, nodes[i].screen_pos, nodes[j].screen_pos, 3)

    # DELETE NODE
    if selected != NONE and keys[pygame.K_BACKSPACE]:
        ADJ_MTX = np.delete(ADJ_MTX, selected, 0)
        ADJ_MTX = np.delete(ADJ_MTX, selected, 1)
        nodes = np.delete(nodes, selected, 0)
        for i in range(selected, NODE_NUM - 1):
            nodes[i].id -= 1
        NODE_NUM -= 1
        selected = NONE

    # DRAW NODES
    for node in nodes:
        node.draw()
        if highlighted == NONE:
            dist = np.linalg.norm(mouse_pos - node.screen_pos)
            if dist < node_size + 1:
                pygame.draw.circle(screen, RED, node.screen_pos, node_size + 2, 2)
                highlighted = node.idx

    if selected != NONE:
        pygame.draw.circle(screen, YELLOW, nodes[selected].screen_pos, node_size + 1, 3)

    if click:
        if highlighted != NONE:
            if mouse_button[0]:
                dragged = highlighted
            elif mouse_button[2]:
                # SELECT HIGHLIGHTED NODE
                if selected == NONE:
                    selected = highlighted
                # UNSELECT NODE
                elif selected == highlighted:
                    selected = NONE
                # CONNECT NODES
                else:
                    ADJ_MTX[selected][highlighted] = 1
                    ADJ_MTX[highlighted][selected] = 1
                    selected = NONE

        # CREATE NEW NODE CONNECTED ONLY TO SELECTED ONE
        elif selected != NONE:
            mouse_real_pos = (np.array(mouse_pos) - SCREEN_CENTER) / zoom + center
            nodes = np.append(nodes, Node(NODE_NUM, mouse_real_pos))
            NODE_NUM += 1
            ADJ_MTX = np.append(ADJ_MTX, np.zeros((1, NODE_NUM - 1)), axis=0)
            ADJ_MTX = np.append(ADJ_MTX, [[0] for i in range(NODE_NUM)], axis=1)
            ADJ_MTX[selected, NODE_NUM - 1] = 1
            ADJ_MTX[NODE_NUM - 1, selected] = 1
            node_dist_mtx = np.zeros((NODE_NUM, NODE_NUM))
            node_vect_mtx = np.zeros((NODE_NUM, NODE_NUM, 2))
            selected = NONE

    # GETS VERY SLOW
    for node in nodes:
        put_text(str(node.idx), screen_pos(node.pos)-np.array([4, 5]), color=WHITE)

    # put_text(str(mouse_pos), (8,8))
    fps_txt = "FPS: " + str(round(clock.get_fps(), 2))
    put_text(fps_txt, (8, 8))
    put_text("NODE REPULSION: " + str(repel_coef), (8, 20))
    put_text("EDGE ELASTICITY: " + str(round(elastic_coef, 3)), (8, 32))
    put_text("ZOOM: " + str(round(zoom, 1)), (8, 44))

    pygame.display.update()

pygame.quit()