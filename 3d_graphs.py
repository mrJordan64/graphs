'''
INSTRUCTIONS:
- LEFTCLICK node + DRAG to drag node
- if no nodes selected, RIGHTCLICK node to select it and RIGHTCLICK it again to unselect it
- to select multiple nodes:
    · CTRL+RIGHTCLICK them. also CTRL+RIGHTCLICK a selected node to unselect it
    · if no nodes selected, LEFTCLICK background + DRAG. otherwise, CTRL+LEFTCLICK background + DRAG
- BACKSPACE to delete all selected nodes (and all their edges)
- RIGHTCLICK another node to connect selected nodes to it
- LEFTCLICK background to create a new node connected to selected nodes
- press F to fix selected nodes in their position
- RIGHTCLICK background to unselect any selected nodes
- RIGHTCLICK edge to delete it
- LEFTCLICK edge to add a new node to it's center

- X to toggle between polar and free modes
- polar mode:
    · A/D           rotate around the center of the graph
    · W/S           move FORWARDS/BACKWARDS
    · SPACE         toggle autorotation
- free mode:
    · W/A/S/D       move horizontally
    · SPACE/SHIFT   move UP/DOWN
    · J/L           rotate camera LEFT/RIGHT
'''


import pygame
import numpy as np

# LOOK FOR BUGS IN NODE SELECTION none?
# node draw order by distance
# edit add node func, more general giving only position and use it for adding node to existing one and subdivide func
# repair copy and paste. maybe add functions?
# graph grammarzzzzz

#   WINDOW SPECS
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
HALF_WIDTH = int(WINDOW_WIDTH / 2)
HALF_HEIGHT = int(WINDOW_HEIGHT / 2)
FPS = 30

CAMERA_LINEAR_SPEED = 0.25
CAMERA_ANGULAR_SPEED = 2 * np.pi / 128

#   COLORS
WHITE  = [255, 255, 255]
BLACK  = [0,   0,   0]
GRAY   = [150, 150, 150]
RED    = [255, 0,   0]
GREEN  = [0,   255, 0]
BLUE   = [100, 100, 255]
YELLOW = [255, 255, 0]

# NODE CONSTANTS
NODE_MASS = 1
NODE_SIZE = 80

# PHYSICS CONSTANTS
FRICTION_COEF = 0.92
ELASTIC_COEF = 0.03
REPEL_COEF = 0.1

NONE = -1

np.random.seed(42)


class Graph:
    def __init__(self, nodes, adjMtx):
        self.nodes = nodes
        self.nodeCounter = len(nodes)
        self.nodeIndex = len(nodes)
        self.adjMtx = adjMtx  # adjacency matrix
        self.vectMtx = np.zeros((self.nodeCounter, self.nodeCounter, 3))  # vectors between nodes mtx
        self.distMtx = np.zeros((self.nodeCounter, self.nodeCounter))     # dists between nodes mtx

    def update_matrices(self):  # update distance and node vector matrices
        # indices optimised to avoid repeating calculations and calculating vects and dists from a node to itself
        self.vectMtx = np.zeros((self.nodeCounter, self.nodeCounter, 3))
        # compute vects for upper triangle
        for i in range(len(self.nodes) - 1):
            for j in range(i + 1, len(self.nodes)):
                self.vectMtx[i, j] = self.nodes[j].pos - self.nodes[i].pos

        # lower triangle is negative transposed upper triangle
        self.vectMtx = self.vectMtx - self.vectMtx.transpose((1,0,2))  # transpose only axis 0 and 1
        # dist[i,j] = norm of vect[i,j]
        self.distMtx = np.linalg.norm(g.vectMtx, axis=2)

    def add_node(self, position):
        self.nodes = np.append(self.nodes, Node(self.nodeIndex, position))
        self.nodeCounter += 1
        self.nodeIndex   += 1

        # extend adjacency matrix with row and column of zeros
        self.adjMtx = np.append(self.adjMtx, np.zeros((1, self.nodeCounter - 1)), axis=0)
        self.adjMtx = np.append(self.adjMtx, np.zeros((self.nodeCounter, 1)), axis=1)

        # reset distance and vector matrices with bigger sizes
        self.distMtx = np.zeros((self.nodeCounter, self.nodeCounter))
        self.vectMtx = np.zeros((self.nodeCounter, self.nodeCounter, 3))

    def delete_node(self, idx):
        # get index of node in node list
        listIdx = self.list_pos_by_index(idx)

        # delete row and col of the node in the adjacency mtx
        self.adjMtx = np.delete(self.adjMtx, listIdx, 0)
        self.adjMtx = np.delete(self.adjMtx, listIdx, 1)

        # delete node from node list
        self.nodes = np.delete(self.nodes, listIdx, 0)
        self.nodeCounter -= 1

    # connect two nodes given their indices in the node list
    def connect_nodes(self, listIdx1, listIdx2):
        self.adjMtx[listIdx1, listIdx2] = 1
        self.adjMtx[listIdx2, listIdx1] = 1

    def subdivide_edge(self, i, j):
        # delete i-j edge
        self.adjMtx[i, j] = 0
        self.adjMtx[j, i] = 0

        # create new node in center of old edge
        position = (self.nodes[i].pos + self.nodes[j].pos) / 2
        self.add_node(position)

        # join it to the nodes of old edge
        self.connect_nodes(i, self.nodeCounter - 1)
        self.connect_nodes(self.nodeCounter - 1, j)

    # get mean position of all nodes
    def center_position(self):
        position = np.zeros(3)
        for node in self.nodes:
            position = position + node.pos

        return position / self.nodeCounter

    # given the unique index of a node, get its position in the nodes list
    def list_pos_by_index(self, index):  ###################################### better name for function?
        for i in range(self.nodeCounter):
            if self.nodes[i].idx == index:
                return i
                break

    # save graph into a .txt
    def save(self):
        filename = "graph_matrices/" + input("Enter graph name: ")
        with open(filename, "w") as f:
            f.write(str(self.nodeCounter) + "\n\n")

            for i in range(self.nodeCounter):
                for j in range(self.nodeCounter):
                    f.write(str(int(self.adjMtx[i, j])) + "  ")
                f.write("\n")


class Node():
    def __init__(self, index, pos, size=NODE_SIZE, color=BLACK, mass=NODE_MASS):
        self.idx = index
        self.pos = pos
        self.screenPos = None
        self.project_to_screen()
        self.speed = np.zeros(3)
        self.fixed = False
        self.selected = False
        self.mass = mass
        self.size = size
        self.color = color
        self.cameraDist = 0

    def update_speed(self, g):
        if not self.fixed:
            index = g.list_pos_by_index(self.idx)
            for i in range(g.nodeCounter):
                # add repelling force from other nodes
                if g.distMtx[index, i] > 0.01:
                    # F = G * (m1*m2)/d^2  ,   F = m * a    therefore   a = G*m2/d^2
                    # the direction of the force is that of the vector between the points, v
                    # adding the normalised vector: a = v/d * G*m2/d^2 = G*v*m2/d^3
                    self.speed -= REPEL_COEF * g.vectMtx[index, i] * g.nodes[i].mass / (g.distMtx[index, i] ** 3)
                # if nodes are connected, add attractive force of the edge
                if g.adjMtx[index, i]:
                    self.speed += ELASTIC_COEF * g.vectMtx[index, i]

        self.speed *= FRICTION_COEF

    def update_size(self, cameraPosition):
        self.cameraDist = np.linalg.norm(cameraPosition - self.pos)
        if self.cameraDist > 0.1:
            self.size = int(NODE_SIZE/self.cameraDist)

    def draw(self):
        if self.fixed:
            pygame.draw.circle(screen, GRAY, self.screenPos, self.size)
        else:
            pygame.draw.circle(screen, self.color, self.screenPos, self.size)
        if self.selected:
            pygame.draw.circle(screen, YELLOW, self.screenPos, self.size + 1, 3)

    def project_to_screen(self):
        # vector from camera position to node
        projVect = self.pos - camera.pos
        k = np.dot(camera.direction, projVect)
        if k < 0.01:
            return False
        # camera is at distance 1 from plane of projection
        # projvect[1]/k is y component of the intersection point of projvect and projection plane
        y = int(HALF_HEIGHT * (1 - projVect[1] / k))
        # projVect[0] * camera.v[2] - projVect[2] * camera.v[0] is the dot product between projvect's horizontal components
        # and also a vector perpendicular to the camera vector
        x = int(HALF_WIDTH * (1 + (projVect[0] * camera.direction[2] - projVect[2] * camera.direction[0]) / k))

        self.screenPos = np.array([x, y])


class Camera(object):
    def __init__(self, point, angle):
        self.pos = point
        self.angle = angle
        self.direction = np.array([np.cos(self.angle), 0, np.sin(self.angle)])
        self.speed = CAMERA_LINEAR_SPEED
        self.angular_speed = CAMERA_ANGULAR_SPEED
        self.distanceToCenter = 8
        self.rotating = False

    def move_free(self, keys):
        self.angle += (keys[pygame.K_j] - keys[pygame.K_l]) * self.angular_speed
        self.direction = np.array([np.cos(self.angle), 0, np.sin(self.angle)])

        vect = np.zeros(3)
        vect += (keys[pygame.K_d] - keys[pygame.K_a]) * \
                np.array([np.cos(self.angle - np.pi / 2), 0, np.sin(self.angle - np.pi / 2)])
        vect += (keys[pygame.K_w] - keys[pygame.K_s]) * self.direction
        vect += (keys[pygame.K_SPACE] - keys[pygame.K_LSHIFT]) * np.array([0, 1, 0])

        self.pos += vect * self.speed

    def move_polar(self, keys, center):
        if self.rotating:
            self.angle += self.angular_speed
        else:
            self.angle += self.angular_speed * (keys[pygame.K_d] - keys[pygame.K_a])

        self.distanceToCenter += (keys[pygame.K_s] - keys[pygame.K_w]) * self.speed
        self.direction = np.array([np.cos(self.angle), 0, np.sin(self.angle)])
        self.pos = center + self.distanceToCenter * np.array([np.cos(self.angle - np.pi), 0, np.sin(self.angle - np.pi)])


def get_graph(filename):
    with open(filename, "r") as f:
        size = int(f.readline().strip("\n"), 10)
        mtx = np.zeros((size, size))

        for i in range(size):
            j = 0
            while j < size:
                char = f.read(1)
                if char != " " and char != "\n":
                    if char == "1":
                        mtx[i, j] = 1
                    j += 1

    return mtx


# Generate a graph using the Erdos-Renyi model
def get_random_graph(size, p):
    mtx = np.zeros((size, size))
    for i in range(size):
        for j in range(i+1, size):
            if np.random.rand() <= p:
                mtx[i, j] = 1
                mtx[j, i] = 1

    return mtx


def draw_line(p1, p2, color=BLACK, thickness=3):
    pygame.draw.line(screen, color, p1.screenPos, p2.screenPos, thickness)


def put_text(text, size, color, place):
    txt = pygame.font.SysFont('minecraftiaregular', size).render(text, True, color)
    screen.blit(txt, place)


# returns true if point is in an ellipse around the nodes and not closer than NODE_SIZE to one of them
def pt_near_edge(point, node1, node2):
    ellipse_constant = 2
    d1 = np.linalg.norm(point - node1.screenPos)
    d2 = np.linalg.norm(point - node2.screenPos)
    d = np.linalg.norm(node1.screenPos - node2.screenPos)

    if d1 + d2 - d > ellipse_constant or d1 < node1.size or d2 < node2.size:
        return False
    else:
        return True


# put 2d mouse position onto 3d space.
# following the line that connects the mouse position on the screen and the camera position,
# project mouse position onto the unique plane parallel to the screen that contains the reference point
# (if i remember correctly :D )
def mouse_real_pos(mouse_pos, ref_pt, camera):
    projVect = ref_pt - camera.pos
    k = np.dot(camera.direction, projVect)
    mousePosPlane = np.multiply([(mouse_pos[0] - HALF_WIDTH) * camera.direction[2],
                                 HALF_HEIGHT - mouse_pos[1],
                                 -(mouse_pos[0] - HALF_WIDTH) * camera.direction[0]],
                                [1 / HALF_WIDTH, 1 / HALF_HEIGHT, 1 / HALF_WIDTH])

    return camera.pos + (camera.direction + mousePosPlane) * k


def random_walk(walk, counter, g):
    walk[3] -= 1
    if walk[2] == 0:
        # if walk in node
        walk[3] -= 2
    if walk[3] < 0:
        walk[3] = counter
        if walk[2] == 1:
            # if walk in edge
            walk[0] = walk[1]
            neighbors = [i for i in range(g.nodeCounter) if g.adjMtx[walk[0], i] == 1]
            walk[1] = np.random.choice(neighbors)
            walk[2] = 0
        else:
            walk[2] = 1
        #walk[2] = (walk[2]+1) % 2

    return walk


# walk = [i,j,state,counter]
counter = 14
walk = [0, 0, 1, 1]

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('3D GRAPHS')
clock = pygame.time.Clock()

#adjMtx = get_graph("graph_matrices/star-cycle")
adjMtx = get_random_graph(15, 0.18)

camera = Camera(np.array([-8., 0., 0.]), 0.)

#nodes = [Node(i, np.random.rand(3) * 5, color=BLACK) for i in range(len(adjMtx))]
# random colors
nodes = [Node(i, np.random.rand(3) * 5, color=np.random.randint(0, 256, 3)) for i in range(len(adjMtx))]

g = Graph(nodes, adjMtx)
center = g.center_position()

nodesByDist = [i for i in range(len(adjMtx))]

# for code convenience, the mouse cursor will be considered a node not related to the graph
mouseCursor = Node(0, np.zeros(3))

click = [False, False]
drag = False
draggedNode   = NONE
selectRectPt  = NONE
selectRect    = NONE
selectedNodes = []
copyAdjMtx    = NONE
show_indexes  = False
edgeEdited    = False
nodeHighlighted = False
mode = 0  # mode 1: free movement, mode 0: rotate around center of graph

###############
x = 0
beat = False
###############

game_over = False
while not game_over:
    clock.tick(FPS)
    screen.fill(WHITE)
    keys = pygame.key.get_pressed()
    mouse_button = (0, 0, 0)
    mouseCursor.screenPos = np.array(pygame.mouse.get_pos())
    speed = []
    click = [False, False]
    edgeEdited = False
    nodeHighlighted = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_button = pygame.mouse.get_pressed()
            if mouse_button[0]:
                click[0] = True
                drag = True
            if mouse_button[2]:
                click[1] = True
        if event.type == pygame.MOUSEBUTTONUP:
            draggedNode = NONE
            if drag:
                drag = False
                if not (keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]):
                    selectedNodes = []
                for node in g.nodes:
                    # if node is in screen and in rectangle
                    if node.screenPos is not False and selectRect.collidepoint(node.screenPos):
                        selectedNodes.append(node.idx)
                        node.selected = True

        if event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            if event.key == pygame.K_c:
                if keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]:
                    print("copy made\n")
                    copyAdjMtx = g.adjMtx
                    print(copyAdjMtx is g.adjMtx)
                    unselectedNodes = [node.idx for node in g.nodes if node.idx not in selectedNodes]
                    unselectedNodes.sort(key=lambda x: g.list_pos_by_index(x), reverse=True)
                    for i in unselectedNodes:
                        _, copyAdjMtx = deleteNode(i, g.nodes, copyAdjMtx) ###############################################################3
            if event.key == pygame.K_i:
                show_indexes = not show_indexes
            if event.key == pygame.K_s:
                if mods & pygame.KMOD_LSHIFT or mods & pygame.KMOD_CAPS:
                    g.save()
            if event.key == pygame.K_v:
                if keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]:
                    g.adjMtx = np.block([[g.adjMtx, np.zeros((len(g.adjMtx), len(copyAdjMtx)), dtype=int)],
                                        [np.zeros((len(copyAdjMtx), len(g.adjMtx)), dtype=int), copyAdjMtx]])
                    for i in range(len(copyAdjMtx)):
                        g.nodes.append(Node(NODE_NUM+i, np.random.rand(3) * 1))

                    g.nodeCounter += len(copyAdjMtx)
                    g.nodeIndex += len(copyAdjMtx)
                    g.distMtx = np.zeros((g.nodeCounter, g.nodeCounter))
                    g.vectMtx = np.zeros((g.nodeCounter, g.nodeCounter, 3))
            if event.key == pygame.K_x:
                mode = (mode+1) % 2
            if event.key == pygame.K_SPACE:
                if mode == 0:
                    camera.rotating = not camera.rotating
            if event.key == pygame.K_b:
                beat = not beat

#############################################################
    if beat:
        x = (x+1/50) % 1
        ELASTIC_COEF = 0.03 + 0.02*np.cos(2*np.pi*x)

    #walk = randomWalk(walk, counter, g)
#############################################################

    #  UPDATE POSITION AND DISTANCE MATRICES
    g.update_matrices()

    # UPDATE SIZES, SPEEDS AND POSITIONS OF NODES
    if draggedNode != NONE:
        index = g.list_pos_by_index(draggedNode)
        mouseCursor.pos = mouse_real_pos(mouseCursor.screenPos, g.nodes[index].pos, camera)
        pygame.draw.circle(screen, RED, mouseCursor.screenPos, g.nodes[index].size - 1, 2)
        g.nodes[index].speed += (mouseCursor.pos - g.nodes[index].pos) * 0.1 * FRICTION_COEF

    #############################################################
    # GRAVITY
    '''
    for i in range(12):
        for j in range(12):
            pos = projectPoint(np.array([i/2, 0, j/2]))
            if pos is not False:
                pygame.draw.circle(screen, BLACK, pos, 1)

    for node in g.nodes:
        node.speed += np.array([0., -0.02, 0.])
        node.updateSize(camera.pos)
        node.updateSpeed(g.nodes)
        node.pos += node.speed
        if node.pos[1] < 0:
            node.pos[1] = 0
        node.screen_pos = projectPoint(node.pos)'''
    #############################################################

    for node in g.nodes:
        node.update_speed(g)
        node.pos += node.speed
        node.update_size(camera.pos)
        node.project_to_screen()

    # DRAW AND EDIT EDGES
    for i in range(g.nodeCounter):
        for j in range(i, g.nodeCounter):
            # if nodes joined and in screen
            if g.adjMtx[i, j] and g.nodes[i].screenPos is not False and g.nodes[j].screenPos is not False:
                if pt_near_edge(mouseCursor.screenPos, g.nodes[i], g.nodes[j]):
                    # draw edge highlighted if mouse near it
                    draw_line(g.nodes[i], g.nodes[j], RED, 5)
                    if mouse_button[2]:  # right click
                        # delete edge
                        g.adjMtx[i, j] = 0
                        g.adjMtx[j, i] = 0
                    elif click[0] and not edgeEdited:  # if left click and no edge edited yet
                        # subdivide edge
                        g.subdivide_edge(i, j)
                        edgeEdited = True
                # if mouse not near edge
                else:
                    # draw edge normally
                    draw_line(g.nodes[i], g.nodes[j], BLACK)
                    '''
                    if i == walk[0] and j == walk[1] and walk[2] == 1:
                        draw_line(g.nodes[i], g.nodes[j], BLUE, 8)'''

    #  DRAW NODES
    nodesByDist = sorted([i for i in range(g.nodeCounter)], reverse=True, key=lambda n: g.nodes[n].cameraDist)
    for i in nodesByDist:
        # if node in screen
        if g.nodes[i].screenPos is not False:
            g.nodes[i].draw()
            '''
            if g.listPosByIndex(node.idx) == walk[0] and walk[2] == 0:
                pygame.draw.circle(screen, BLUE, node.screen_pos, node.size + 1)'''
            mouseDist = np.linalg.norm(mouseCursor.screenPos - g.nodes[i].screenPos)
            # if mouse over node
            if mouseDist < g.nodes[i].size + 1:
                # draw highlight
                nodeHighlighted = True
                if click[0]:  # left click
                    draggedNode = g.nodes[i].idx
                pygame.draw.circle(screen, RED, g.nodes[i].screenPos, g.nodes[i].size + 1, 2)
                if click[1]:  # right click
                    if not selectedNodes: # if no selected nodes, select it
                        g.nodes[i].selected = True
                        selectedNodes.append(g.nodes[i].idx)
                    else:
                        if keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]:
                            # if there are selected nodes and control pressed, add or remove from selected nodes
                            g.nodes[i].selected = not g.nodes[i].selected
                            if g.nodes[i].selected:
                                selectedNodes.append(g.nodes[i].idx)
                            else:
                                selectedNodes.remove(g.nodes[i].idx)
                        else:
                            # if control not pressed, connect selected nodes to node
                            listPos = g.list_pos_by_index(g.nodes[i].idx)
                            g.nodes[i].selected = False
                            for index in selectedNodes:
                                listPos2 = g.list_pos_by_index(index)
                                g.connect_nodes(listPos, listPos2)
                                g.nodes[listPos2].selected = False
                            selectedNodes = []

    # MOUSE RECTANGLE SELECTION
    # if click[0], save mouse position to create the selection rectangle in case mouse is dragged
    if click[0]:
        selectRectPt = mouseCursor.screenPos
        selectRect = pygame.Rect(selectRectPt, [0, 0])
        if nodeHighlighted:
            drag = False
    if drag:
        # determine up-left and down-right pts of selection rectangle
        up_left = np.array([min(selectRectPt[0], mouseCursor.screenPos[0]),
                            min(selectRectPt[1], mouseCursor.screenPos[1])])
        down_right = np.array([max(selectRectPt[0], mouseCursor.screenPos[0]),
                               max(selectRectPt[1], mouseCursor.screenPos[1])])

        # get selection rectangle and draw it
        selectRect = pygame.Rect(up_left, [down_right[0]-up_left[0], down_right[1]-up_left[1]])
        pygame.draw.rect(screen, BLACK, selectRect, width=1)
        # highlight nodes in selection rectangle
        for node in g.nodes:
            # if node is in rectangle
            if node.screenPos is not False and selectRect.collidepoint(node.screenPos):
                pygame.draw.circle(screen, RED, node.screenPos, node.size + 1, 2)

    # if right click no node or edge, unselect all nodes
    if click[1] and not nodeHighlighted and not edgeEdited:
        for node in selectedNodes:
            g.nodes[g.list_pos_by_index(node)].selected = False
        selectedNodes = []

    if selectedNodes:
        # ADD NODE AND JOIN IT TO SELECTED ONES
        if click[0] and not (nodeHighlighted or edgeEdited or keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]):
            listPos = g.list_pos_by_index(selectedNodes[0])
            g.nodes[listPos].selected = False
            mouseCursor.pos = mouse_real_pos(mouseCursor.screenPos, g.nodes[listPos].pos, camera)
            g.add_node(mouseCursor.pos)
            g.connect_nodes(listPos, g.nodeCounter - 1)
            for node in selectedNodes[1:]:
                listPos = g.list_pos_by_index(node)
                g.nodes[listPos].selected = False
                g.connect_nodes(listPos, g.nodeCounter - 1)
            selected_nodes = []
        # DELETE NODE
        if keys[pygame.K_BACKSPACE]:
            for node in selectedNodes:
                g.delete_node(node)
            selectedNodes = []
        # FIX NODE
        if keys[pygame.K_f]:
            for node in selectedNodes:
                listPos = g.list_pos_by_index(node)
                g.nodes[listPos].selected = False
                g.nodes[listPos].fixed = not g.nodes[listPos].fixed
            selectedNodes = []

    if mode == 1:
        camera.move_free(keys)
        put_text("MODE: FREE", 16, BLACK, (8, 24))
    elif mode == 0:
        camera.move_polar(keys, g.center_position())
        put_text("MODE: POLAR", 16, BLACK, (8, 24))

    put_text("FPS: " + str(round(clock.get_fps(), 5)), 16, BLACK, (8, 8))
    # put_text(str(camera.pos), 16, BLACK, (8, 20))

    if show_indexes:
        for node in g.nodes:
            if node.screenPos is not False:
                put_text(str(node.idx), 12, WHITE, node.screenPos - np.array([4, 5]))

    pygame.display.update()

pygame.quit()
quit()