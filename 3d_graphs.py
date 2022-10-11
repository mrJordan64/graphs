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

# TODO:
# rewrite instructions!
# change screen_color function to a look-up table?
# clean up get adj mtx function
# bugs: none?
# function to calculate if node or edge is closer to the camera. written down on brown paper!
# graph grammarzzzzz

#   WINDOW SPECS
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
HALF_WIDTH = int(WINDOW_WIDTH / 2)
HALF_HEIGHT = int(WINDOW_HEIGHT / 2)
FPS = 30

TEXT_SIZE = 16
TEXT_SPACING = 16

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

BACKGROUND_COLOR = np.array(WHITE)
START_FADE_DIST = 7
END_FADE_DIST = 12

# NODE CONSTANTS
NODE_MASS = 1
NODE_SIZE = 80

# PHYSICS CONSTANTS
FRICTION_COEF  = 0.92
ELASTIC_COEF   = 0.01
REPELLING_COEF = 0.02

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

    def update_nodes(self):
        for node in self.nodes:
            node.update_speed()
            node.update_position(camera)
            node.update_size()

    def add_node(self, position):
        self.nodes.append(Node(self.nodeIndex, position))
        self.nodeCounter += 1
        self.nodeIndex   += 1

        # extend adjacency matrix with row and column of zeros
        self.adjMtx = np.append(self.adjMtx, np.zeros((1, self.nodeCounter - 1)), axis=0)
        self.adjMtx = np.append(self.adjMtx, np.zeros((self.nodeCounter, 1)), axis=1)

        # reset distance and vector matrices with bigger sizes #######################################################
        self.distMtx = np.zeros((self.nodeCounter, self.nodeCounter))
        self.vectMtx = np.zeros((self.nodeCounter, self.nodeCounter, 3))

    def delete_nodes(self, node_ids):
        # get index of node
        node_indices = [self.index_by_id(id) for id in node_ids]

        # delete row and col of the node in the adjacency mtx
        self.adjMtx = np.delete(self.adjMtx, node_indices, 0)
        self.adjMtx = np.delete(self.adjMtx, node_indices, 1)

        # delete node from node list
        self.nodes = np.delete(self.nodes, node_indices, 0)
        self.nodeCounter = len(self.nodes)

    # connect two nodes given their indices in the node list
    def connect_node_to(self, nodeId, nodeIds):
        nodeIdx = g.index_by_id(nodeId)
        for index in [g.index_by_id(id) for id in nodeIds]:
            if index != nodeIdx:
                self.adjMtx[nodeIdx, index] = 1
                self.adjMtx[index, nodeIdx] = 1

    # get mean position of all nodes
    def center_position(self):
        position = np.zeros(3)
        for node in self.nodes:
            position = position + node.pos

        return position / self.nodeCounter

    # given the unique index of a node, get its position in the nodes list
    def index_by_id(self, index):  ###################################### better name for function?
        for i in range(self.nodeCounter):
            if self.nodes[i].id == index:
                return i

    # save graph into a .txt
    def save(self):
        filename = "graph_matrices/" + input("Enter graph name: ")
        with open(filename, "w") as f:
            f.write(str(self.nodeCounter) + "\n\n")

            for i in range(self.nodeCounter):
                for j in range(self.nodeCounter):
                    f.write(str(int(self.adjMtx[i, j])) + "  ")
                f.write("\n")

    def draw(self, showIds):
        # nodes will be drawn from back to front
        # first sort them by dist to camera
        nodeIdxsByDistance = sorted(range(self.nodeCounter), reverse=True, key=lambda n: self.nodes[n].cameraDist)
        for i in nodeIdxsByDistance:
            # if not in screen, skip node
            if not self.nodes[i].inScreen:
                continue

            # else, draw node normally
            self.nodes[i].draw()
            # add highlight or selection if necessary
            self.nodes[i].draw_mods()

            # draw edges between node j and its neighbors closer to the camera
            for j in nodeIdxsByDistance[nodeIdxsByDistance.index(i) + 1:]:
                if self.adjMtx[i, j] and self.nodes[j].inScreen:
                    self.draw_edge(i, j)

            # draw node but smaller to oclude part of the drawn edge
            self.nodes[i].draw(modifier=-4)

            if showIds:
                # write node's ID on it
                put_text(str(self.nodes[i].id), 12, self.nodes[i].screenPos - np.array([4, 5]), WHITE)

    def draw_edge(self, i, j):
        # get aprox dist and avg color to generate color of edge
        avgDistance = ( self.nodes[i].cameraDist + self.nodes[j].cameraDist ) / 2
        if avgDistance > END_FADE_DIST:
            return
        avgColor = ( self.nodes[i].color + self.nodes[j].color ) / 2

        # draw edge normally
        pygame.draw.line(screen, screen_color(avgColor, avgDistance), self.nodes[i].screenPos, self.nodes[j].screenPos, int(24 / avgDistance))
        # highlight edge
        if edgesUnderMouse[i, j]:
            pygame.draw.line(screen, screen_color(RED, avgDistance), self.nodes[i].screenPos, self.nodes[j].screenPos, int(40 / avgDistance))

    def subdivide_edges(self, edgeMtx):
        for i in range(g.nodeCounter):
            for j in range(i+1, g.nodeCounter):
                if edgeMtx[i, j]:
                    # delete i-j edge
                    self.adjMtx[i, j], self.adjMtx[j, i] = 0, 0
                    # create new node in center of old edge
                    self.add_node((self.nodes[i].pos + self.nodes[j].pos) / 2)
                    # join it to the nodes of old edge
                    self.connect_node_to(self.nodes[-1].id, [self.nodes[i].id, self.nodes[j].id])
                    # uncomment if we only want to edit one edge per click
                    # break

    def delete_edges(self, edgeMtx):
        for i in range(g.nodeCounter):
            for j in range(i+1, g.nodeCounter):
                if edgeMtx[i, j]:
                    self.adjMtx[i, j], self.adjMtx[j, i] = 0, 0
                    # uncomment if we only want to edit one edge per click
                    # break

    def selected_subgraph(self):
        # start with whole adjMtx
        selectedMtx = self.adjMtx
        # get unselected nodes and remove them from copy
        unselectedNodes = [i for i in range(self.nodeCounter) if i not in selectedNodes]
        # delete rows and cols of unselected nodes in the copy
        selectedMtx = np.delete(selectedMtx, unselectedNodes, 0)
        selectedMtx = np.delete(selectedMtx, unselectedNodes, 1)

        return selectedMtx

    def add_subgraph(self, copyMtx):
        # add unconnected nodes
        for i in range(len(copyMtx)):
            self.add_node(np.random.rand(3) * 1)
        # adjMtx now bigger but need to connect new nodes

        # extend copyMtx with 0s as [[A, B], [C, D]] with A, B, C mtx of 0s and D copyMtx
        A = np.zeros((self.nodeCounter - len(copyMtx), self.nodeCounter - len(copyMtx)), dtype=int)
        B = np.zeros((self.nodeCounter - len(copyMtx), len(copyMtx)), dtype=int)
        C = np.zeros((len(copyMtx), self.nodeCounter - len(copyMtx)), dtype=int)
        D = copyMtx

        copyMtx = np.block([[A, B],
                            [C, D]])

        # add extended copyMtx to g.adjMtx
        self.adjMtx = self.adjMtx + copyMtx


class Node():
    def __init__(self, id, pos, size=NODE_SIZE, color=BLACK, mass=NODE_MASS):
        self.id = id
        self.pos = pos
        self.speed = np.zeros(3)
        self.fixed = False
        self.mass = mass
        self.size = size
        self.color = np.array(color)
        self.cameraDist = 0

        self.inScreen = False
        self.screenPos = np.zeros(2)
        self.screenSize = 0
        self.update_screen_pos(camera)
        self.update_size()

    def update_speed(self):
        if not self.fixed:
            index = g.nodes.index(self)

            for i in range(g.nodeCounter):
                # add repelling force from other nodes
                if g.distMtx[index, i] > 0.01:
                    # F = G * (m1*m2)/d^2  ,   F = m * a    therefore   a = G*m2/d^2
                    # the direction of the force is that of the vector between the points, v
                    # adding the normalised vector: a = v/d * G*m2/d^2 = G*v*m2/d^3
                    self.speed -= REPELLING_COEF * g.vectMtx[index, i] * g.nodes[i].mass / (g.distMtx[index, i] ** 3)

                # if nodes are connected, add attraction force
                if g.adjMtx[index, i]:
                    self.speed += ELASTIC_COEF * g.vectMtx[index, i]

        # if node being dragged
        if self.id == draggedNodeId:
            self.speed += ELASTIC_COEF * 10 * (mouse_3d_pos(mousePos, self.pos) - self.pos)

        self.speed *= FRICTION_COEF

    def update_position(self, camera):
        self.pos += self.speed
        self.cameraDist = np.linalg.norm(camera.pos - self.pos)
        self.update_screen_pos(camera)

    def update_size(self):
        if self.cameraDist > 0.1:
            self.screenSize = int(self.size / self.cameraDist)

    def draw(self, modifier=0):
        if self.cameraDist > END_FADE_DIST: return

        if self.fixed:
            pygame.draw.circle(screen, screen_color(GRAY, self.cameraDist), self.screenPos, self.screenSize)
        else:
            pygame.draw.circle(screen, screen_color(self.color, self.cameraDist), self.screenPos, self.screenSize+modifier)

    def draw_mods(self):
        if self.cameraDist > END_FADE_DIST: return

        if self.id in selectedNodes:
            pygame.draw.circle(screen, screen_color(YELLOW, self.cameraDist), self.screenPos, self.screenSize, 3)
        if self.id in highlightedNodes:
            pygame.draw.circle(screen, screen_color(RED, self.cameraDist), self.screenPos, self.screenSize + 1, 2)

    def update_screen_pos(self, camera):
        # vector from camera position to node
        projVect = self.pos - camera.pos
        dot = np.dot(camera.viewVector, projVect)
        if dot < 0.01:
            self.inScreen = False
            return
        else:
            self.inScreen = True

        # camera is at distance 1 from plane of projection
        # projvect[1]/dot is y component of the intersection point of projvect and projection plane
        y = int(HALF_HEIGHT * (1 - projVect[1] / dot))

        # projVect[0] * camera.v[2] - projVect[2] * camera.v[0] is the dot product between projvect's horizontal components
        # and also a vector perpendicular to the camera vector
        x = int(HALF_WIDTH * (1 + (projVect[0] * camera.viewVector[2] - projVect[2] * camera.viewVector[0]) / dot))

        self.screenPos = np.array([x, y])

    def invert_selection(self):
        # add or remove node from selection list
        if self.id in selectedNodes:
            selectedNodes.remove(self.id)
        else:
            selectedNodes.append(self.id)


class Camera(object):
    def __init__(self, point, angle):
        self.pos = point
        self.angle = angle
        self.viewVector = u_vec_xz(angle)

        self.viewMode = "orbital"
        self.distanceToCenter = 8
        self.rotating = False

    def toggle_viewMode(self):
        if self.viewMode == "orbital":
            self.viewMode = "free"
        elif self.viewMode == "free":
            self.viewMode = "orbital"

    def move(self, keys):
        if self.viewMode == "orbital":
            self.move_orbital(keys)
        elif self.viewMode == "free":
            self.move_free(keys)

    def move_free(self, keys):
        # rotate left-right with J-L
        self.angle += CAMERA_ANGULAR_SPEED * (keys[pygame.K_j] - keys[pygame.K_l])
        # update view vector
        self.viewVector = np.array([np.cos(self.angle), 0, np.sin(self.angle)])
        # move left-right with A-D
        vect = u_vec_xz(self.angle - np.pi/2) * (keys[pygame.K_d] - keys[pygame.K_a])
        # move forward-backward with W-S
        vect += self.viewVector * (keys[pygame.K_w] - keys[pygame.K_s])
        # move up-down with SPACE-SHIFT
        vect += np.array([0, 1, 0]) * (keys[pygame.K_SPACE] - keys[pygame.K_LSHIFT])

        self.pos += vect * CAMERA_LINEAR_SPEED

    def move_orbital(self, keys):
        # rotate
        if self.rotating:
            self.angle += CAMERA_ANGULAR_SPEED
        else:
            self.angle += CAMERA_ANGULAR_SPEED * (keys[pygame.K_d] - keys[pygame.K_a])

        # update view vector
        self.viewVector = u_vec_xz(self.angle)
        # move forward-backwards with W-S
        self.distanceToCenter += CAMERA_LINEAR_SPEED * (keys[pygame.K_s] - keys[pygame.K_w])
        # position is at circle centered on the graph, radius distToCenter and angle+180º to be looking inside
        self.pos = g.center_position() + self.distanceToCenter * u_vec_xz(self.angle + np.pi)


def get_adj_mtx(filename):
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
def get_random_graph(n, p):
    adjMtx = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if np.random.rand() <= p:
                adjMtx[i, j] = 1
                adjMtx[j, i] = 1

    return adjMtx


def put_text(text, size, position, color=BLACK):
    txt = pygame.font.SysFont('minecraftiaregular', size).render(text, True, color)
    screen.blit(txt, position)


# returns true if point is in an ellipse around the nodes and not closer than NODE_SIZE to one of them
def pt_near_segment(point, node1, node2):
    ellipseConstant = 0.5
    d1 = np.linalg.norm(point - node1.screenPos)
    d2 = np.linalg.norm(point - node2.screenPos)
    d = np.linalg.norm(node1.screenPos - node2.screenPos)

    if d1 + d2 > d + ellipseConstant or d1 < node1.screenSize or d2 < node2.screenSize:
        return False
    else:
        return True


# fades the color of an object when it's far away
def screen_color(realColor, distance):
    if distance <= START_FADE_DIST: return realColor
    elif distance >= END_FADE_DIST: return BACKGROUND_COLOR

    # eq of a line that has values realColor at distance=startFade and background color at x=endFade
    return realColor + (distance - START_FADE_DIST)*(BACKGROUND_COLOR - realColor)/(END_FADE_DIST - START_FADE_DIST)


# put 2d mouse position onto 3d space.
# following the line that connects the mouse position on the screen and the camera position,
# project mouse position onto the unique plane parallel to the screen that contains the reference point
# (if i remember correctly :D )
def mouse_3d_pos(mousePos, ref_pt):
    projVect = ref_pt - camera.pos
    k = np.dot(camera.viewVector, projVect)
    mousePosPlane = np.multiply([(mousePos[0] - HALF_WIDTH) * camera.viewVector[2], HALF_HEIGHT - mousePos[1],
                                 -(mousePos[0] - HALF_WIDTH) * camera.viewVector[0]], [1 / HALF_WIDTH, 1 / HALF_HEIGHT, 1 / HALF_WIDTH])

    return camera.pos + (camera.viewVector + mousePosPlane) * k


# gets rectangle generated by startPt and mousePos point
def get_selection_rectangle(startPt, mousePos):
    # get up-left and down-right pts of selection rectangle
    upLeftPt    = [min(startPt[0], mousePos[0]), min(startPt[1], mousePos[1])]
    downRightPt = [max(startPt[0], mousePos[0]), max(startPt[1], mousePos[1])]

    # get width and height of rectangle
    width   = downRightPt[0] - upLeftPt[0]
    height  = downRightPt[1] - upLeftPt[1]

    return pygame.Rect(upLeftPt, [width, height])


# return list of ids of nodes in the given rectangle
def nodes_in_rectangle_ids(rect):
    return [node.id for node in g.nodes if node.inScreen and rect.collidepoint(node.screenPos)]


# unitary vector in plane xz given angle
def u_vec_xz(angle):
    return np.array([np.cos(angle), 0, np.sin(angle)])


def nodes_under_mouse():
    _mouseOverNodes = False
    closestNodeIndex = False
    closestCameraDist = 10000

    for i in range(g.nodeCounter):
        distToMouse = np.linalg.norm(mousePos - g.nodes[i].screenPos)
        # if mouse over node
        if distToMouse < g.nodes[i].screenSize + 1 and g.nodes[i].cameraDist < closestCameraDist:
            closestNodeIndex = g.nodes[i].id
            closestCameraDist = g.nodes[i].cameraDist
            _mouseOverNodes = True

    return _mouseOverNodes, closestNodeIndex

# for every edge check if mouse over it
def edges_under_mouse():
    highlightedEdges = np.zeros((g.nodeCounter, g.nodeCounter))
    _mouseOverEdges = False

    for i in range(g.nodeCounter):
        for j in range(i+1, g.nodeCounter):
            if g.adjMtx[i,j] and g.nodes[i].inScreen and g.nodes[j].inScreen:
                if pt_near_segment(mousePos, g.nodes[i], g.nodes[j]):
                    highlightedEdges[i, j] = 1
                    highlightedEdges[j, i] = 1
                    _mouseOverEdges = True

    return _mouseOverEdges, highlightedEdges

#############################################
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

x = 0
beat = False

# END OF FUNCTIONS #####################################################################################################################################

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('3D GRAPHS')
clock = pygame.time.Clock()

#adjMtx = get_adj_mtx("graph_matrices/star-cycle")
adjMtx = get_random_graph(100, 0.035)

camera = Camera(np.array([-8., 0., 0.]), 0.)

nodes = [Node(i, np.random.rand(3) * 5) for i in range(len(adjMtx))]
# random colors
# nodes = [Node(i, np.random.rand(3) * 5, color=np.random.randint(0, 256, 3)) for i in range(len(adjMtx))]

g = Graph(nodes, adjMtx)
center = g.center_position()

nodesByDist = [i for i in range(len(adjMtx))]

selectedNodes      = []
draggedNodeId      = None
selectedSubGraph   = None
rectangleSelection = False
showIds            = False

g.update_matrices()
g.update_nodes()

stop = False
while not stop:
    clock.tick(FPS)
    screen.fill(BACKGROUND_COLOR)

    ##############################
    leftClick     = False
    rightClick    = False
    mouseButtonUp = False
    mousePos = np.array(pygame.mouse.get_pos())

    keys = pygame.key.get_pressed()
    ctrlPressed = keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]
    camera.move(keys)

    highlightedNodes = []
    ##############################

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stop = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:  # if leftclick
                leftClick = True
            if pygame.mouse.get_pressed()[2]:  # if rightclick
                rightClick = True

        if event.type == pygame.MOUSEBUTTONUP:
            mouseButtonUp = True

        if event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()  # for CAPS and SHIFT

            # CTRL+C -> copy inducted subgraph of selected nodes
            if event.key == pygame.K_c and ctrlPressed:
                selectedSubGraph = g.selected_subgraph()

            # CTRL+V -> paste inducted subgraph of selected nodes
            if event.key == pygame.K_v and ctrlPressed and selectedSubGraph is not None:
                g.add_subgraph(selectedSubGraph)

            # i -> toggle showing graph indices
            if event.key == pygame.K_i:
                showIds = not showIds

            # capital S -> save graph into file
            if event.key == pygame.K_s and (mods & pygame.KMOD_LSHIFT or mods & pygame.KMOD_CAPS):
                g.save()

            # x -> toggle view mode
            if event.key == pygame.K_x:
                camera.toggle_viewMode()

            # SPACE -> toggle camera rotation around graph
            if event.key == pygame.K_SPACE:
                camera.rotating = not camera.rotating

            # backspace -> delete selected nodes
            if event.key == pygame.K_BACKSPACE:
                g.delete_nodes(selectedNodes)
                selectedNodes = []

            # f key -> fix or unfix selected nodes
            if event.key == pygame.K_f:
                indices = [g.index_by_id(id) for id in selectedNodes]
                for index in indices:
                    g.nodes[index].fixed = not g.nodes[index].fixed

            '''
            # b -> toggle beat
            if event.key == pygame.K_b:
                beat = not beat
            '''

    # check if mouse is over any nodes
    mouseOverNodes, nodeUnderMouseId= nodes_under_mouse()
    mouseOverEdges, edgesUnderMouse = edges_under_mouse()
    mouseOverSomething = mouseOverNodes or mouseOverEdges

    if mouseOverNodes:
        highlightedNodes.append(nodeUnderMouseId)

    if mouseOverEdges:
        if leftClick:
            g.subdivide_edges(edgesUnderMouse)
        if rightClick:
            g.delete_edges(edgesUnderMouse)

    if mouseButtonUp:
        # stop rectangle selection and node dragging
        draggedNodeId = None

        # if a selection rectangle was being made, invert selection of nodes inside it
        if rectangleSelection:
            for id in nodes_in_rectangle_ids(selectionRect):
                index = g.index_by_id(id)
                g.nodes[index].invert_selection()

            rectangleSelection = False

    # leftclick node -> start dragging it
    if leftClick and mouseOverNodes:
        draggedNodeId = nodeUnderMouseId

    # leftclick + NOT nodes under mouse + selected nodes + CTRL NOT pressed -> unselect everything
    if leftClick and not mouseOverNodes and selectedNodes and not ctrlPressed:
        selectedNodes = []

    # leftclick + NOT nodes under mouse +
    # + ( NOT selected nodes OR selected nodes + CTRL pressed ) -> start selection rectangle
    if leftClick and not mouseOverNodes and (not selectedNodes or (selectedNodes and ctrlPressed)):
        rectangleSelection = True
        startRectPoint = mousePos
        selectionRect = pygame.Rect(startRectPoint, [0, 0])

    # rightclick node + (NOT selected nodes OR ( selected nodes + control pressed ))-> invert selection of node
    if rightClick and mouseOverNodes:
        if not selectedNodes or (selectedNodes and ( len(selectedNodes) == 1 or ctrlPressed )):
            index = g.index_by_id(nodeUnderMouseId)
            g.nodes[index].invert_selection()

    # rightclick node + selected nodes + NOT control pressed -> connect selected nodes to node
    if rightClick and mouseOverNodes and selectedNodes and not ctrlPressed:
        g.connect_node_to(nodeUnderMouseId, selectedNodes)

    # rightClick + NOT nodes under mouse + selected nodes -> connect selected to new node
    if rightClick and not mouseOverNodes and selectedNodes:
        # add new node at mouse position using first selected node as reference for distance
        index = g.index_by_id(selectedNodes[0])
        g.add_node(mouse_3d_pos(mousePos, g.nodes[index].pos))

        # connect node to selected nodes
        g.connect_node_to(g.nodes[-1].id, selectedNodes)

    if rectangleSelection:
        selectionRect = get_selection_rectangle(startRectPoint, mousePos)
        # draw selection rectangle
        pygame.draw.rect(screen, RED, selectionRect, width=1)

        # add nodes in selection rectangle to highlighted list
        highlightedNodes = highlightedNodes + nodes_in_rectangle_ids(selectionRect)

    # UPDATE GRAPH AND DRAW IT
    g.update_matrices()
    g.update_nodes()
    g.draw(showIds)

    # WRITE STUFF ON SCREEN
    screen_text = [
        f"FPS: {round(clock.get_fps(), 2)}",
        "CAM MODE: " + camera.viewMode]
        # f"dist 0: {round(g.nodes[0].cameraDist)}",
        # f"pos 0: {np.round(g.nodes[0].pos)}",
        # f"screenpos 0: {np.round(g.nodes[0].screenPos)}",
        # f"node 0 color: {g.nodes[0].color}"]

    for i in range(len(screen_text)):
        put_text(screen_text[i], TEXT_SIZE, (8, 8+TEXT_SPACING*i))

    pygame.display.update()

    # ###################################################################################################
    '''
    if beat:
        x = (x + 1 / 50) % 1
        ELASTIC_COEF = 0.03 + 0.02 * np.cos(2 * np.pi * x)

    # walk = randomWalk(walk, counter, g)
    
    # GRAVITY
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
    ###########################################################################################

pygame.quit()
quit()