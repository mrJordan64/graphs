'''
INSTRUCTIONS:
- LEFTCLICK node + DRAG to drag node
- if no nodes selected, RIGHTCLICK node to select it and RIGHTCLICK it again to unselect it
- to select multiple nodes:
    · CTRL+RIGHTCLICK them. also CTRL+RIGHTCLICK a selected node to unselect it
    · if no nodes selected, LEFTCLICK background + DRAG. otherwise, CTRL+LEFTCLICK background + DRAG
- press BACKSPACE to delete all selected nodes (and all their edges!)
- RIGHTCLICK another node to connect selected nodes to it
- LEFTCLICK background to create a new node connected to selected nodes
- press F to fix selected nodes in their position
- RIGHTCLICK background to unselect any selected nodes
- RIGHTCLICK edge to delete it
- LEFTCLICK edge to add a new node to it's center

- press X to toggle between polar and free modes
- polar mode:
    · press A/D to rotate LEFT/RIGHT around center of graph
    · press W/S to move FORWARDS/BACKWARDS
    · press SPACE to autorotate
- free mode:
    · W/A/S/D to move horizontally
    · SPACE/SHIFT to move UP/DOWN
    · J/L to rotate camera LEFT/RIGHT
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

#   COLORS
WHITE  = [255, 255, 255]
BLACK  = [0,   0,   0]
GRAY   = [150, 150, 150]
RED    = [255, 0,   0]
GREEN  = [0,   255, 0]
BLUE   = [100,   100,   255]
YELLOW = [255, 255, 0]

NODE_MASS = 1
NODE_SIZE = 9
FRICTION_COEF = 0.9
ELASTIC_COEF = 0.03
REPEL_COEF = 0.1

NONE = -1


class Graph:
    def __init__(self, nodes, adjMtx):
        self.nodes = nodes
        self.order = len(nodes)
        self.nodeNum = len(nodes)
        self.adjMtx = adjMtx
        self.distMtx = np.zeros((self.nodeNum, self.nodeNum))
        self.vectMtx = np.zeros((self.nodeNum, self.nodeNum, 3))

    def updateMtxs(self):
        # CALCULATE VECTORS AND DISTANCES BETWEEN NODES
        for i in range(len(self.nodes)):
            for j in range(i, len(self.nodes)):
                self.vectMtx[i, j] = self.nodes[j].pos - self.nodes[i].pos
                self.vectMtx[j, i] = -1 * self.vectMtx[i, j]
                self.distMtx[i, j] = np.linalg.norm(self.vectMtx[i, j])
                self.distMtx[j, i] = self.distMtx[i, j]

    def addNode(self, position):
        self.nodes = np.append(self.nodes, Node(self.nodeNum, position))
        self.order += 1
        self.nodeNum += 1
        self.adjMtx = np.append(self.adjMtx, np.zeros((1, self.order-1)), axis=0)
        self.adjMtx = np.append(self.adjMtx, np.zeros((self.order, 1)), axis=1)
        self.distMtx = np.zeros((self.order, self.order))
        self.vectMtx = np.zeros((self.order, self.order, 3))

    def deleteNode(self, idx):
        listPos = self.listPosByIndex(idx)

        self.adjMtx = np.delete(self.adjMtx, listPos, 0)
        self.adjMtx = np.delete(self.adjMtx, listPos, 1)
        self.nodes = np.delete(self.nodes, listPos, 0)
        self.order -= 1

    def connect(self, listPos1, listPos2):
        self.adjMtx[listPos1, listPos2] = 1
        self.adjMtx[listPos2, listPos1] = 1

    def subdivideEdge(self, i, j):
        # delete edge, create new node in center of old edge and
        # join it to the nodes of old edge
        self.adjMtx[i, j] = 0
        self.adjMtx[j, i] = 0

        position = (self.nodes[i].pos + self.nodes[j].pos) / 2  # new node position
        self.addNode(position)
        self.connect(i, self.order-1)
        self.connect(self.order - 1, j)

    def getCenter(self):
        center = np.array([0, 0, 0])
        for node in self.nodes:
            center = center + node.pos

        return center / len(self.nodes)

    def listPosByIndex(self, index):
        for i in range(len(self.nodes)):
            if self.nodes[i].idx == index:
                return i
                break

    def save(self):
        size = len(self.nodes)
        filename = "graph_matrices/" + input("Enter graph name: ")
        with open(filename, "w") as f:
            f.write(str(size) + "\n\n")

            for i in range(size):
                for j in range(size):
                    f.write(str(int(self.adjMtx[i, j])) + "  ")
                f.write("\n")


class Node():
    def __init__(self, idx, pos, size=NODE_SIZE, color=BLACK, mass=NODE_MASS):
        self.idx = idx
        self.pos = pos
        self.screen_pos = project(self.pos)
        self.speed = np.array([0., 0., 0.])
        self.fixed = False
        self.selected = False
        self.mass = mass
        self.size = size
        self.color = color
        self.cameraDist = 0

    def updateSpeed(self, g):
        if not self.fixed:
            index = g.listPosByIndex(self.idx)
            for i in range(g.order):
                # ADD REPELLING FORCES FROM OTHER NODES
                if g.distMtx[index, i] > 0.01:
                    self.speed -= g.vectMtx[index, i] * REPEL_COEF * self.mass / (g.distMtx[index, i] ** 3)
                # ADD ATTRACTIVE FORCES FROM EDGES
                if g.adjMtx[index, i]:
                    self.speed += g.vectMtx[index, i] * ELASTIC_COEF

        self.speed *= FRICTION_COEF

    def updateSize(self, cameraPosition):
        self.cameraDist = np.linalg.norm(cameraPosition - self.pos)
        if self.cameraDist > 0.1:
            self.size = int(100/self.cameraDist)

    def draw(self):
        if self.fixed:
            pygame.draw.circle(screen, GRAY, self.screen_pos, self.size)
        else:
            pygame.draw.circle(screen, self.color, self.screen_pos, self.size)
        if self.selected:
            pygame.draw.circle(screen, YELLOW, self.screen_pos, self.size + 1, 3)


class Camera(object):
    def __init__(self, point, angle):
        self.pos = point
        self.speed = 0.25
        self.angle = angle
        self.v = np.array([np.cos(self.angle), 0, np.sin(self.angle)])
        self.rotate = False
        self.angular_speed = 2 * np.pi / 128
        self.dist = 8

    def move_free(self, keys):
        self.angle += (keys[pygame.K_j] - keys[pygame.K_l]) * self.angular_speed
        self.v = np.array([np.cos(self.angle), 0, np.sin(self.angle)])

        vect = np.zeros(3)
        vect += (keys[pygame.K_d] - keys[pygame.K_a]) * \
                np.array([np.cos(self.angle - np.pi / 2), 0, np.sin(self.angle - np.pi / 2)])
        vect += (keys[pygame.K_w] - keys[pygame.K_s]) * self.v
        vect += (keys[pygame.K_SPACE] - keys[pygame.K_LSHIFT]) * np.array([0, 1, 0])

        self.pos += vect * self.speed

    def move_around(self, keys, center):
        if self.rotate:
            self.angle += self.angular_speed
        else:
            self.angle += (keys[pygame.K_d] - keys[pygame.K_a]) * self.angular_speed

        self.v = np.array([np.cos(self.angle), 0, np.sin(self.angle)])
        self.dist += (keys[pygame.K_s] - keys[pygame.K_w]) * self.speed
        self.pos = center + self.dist * np.array([np.cos(self.angle - np.pi), 0, np.sin(self.angle - np.pi)])


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


def get_random_graph(size, p):
    mtx = np.zeros((size, size))
    for i in range(size):
        for j in range(i+1, size):
            if np.random.rand() <= p:
                mtx[i, j] = 1
                mtx[j, i] = 1

    return mtx


def project(point):
    # vector from camera position to point
    projVect = point - camera.pos
    k = np.dot(camera.v, projVect)
    if k < 0.01:
        return False
    # camera is at distance 1 from plane of projection
    # projvect[1]/k is y component of the intersection point of projvect and projection plane
    y = int(HALF_HEIGHT * (1 - projVect[1] / k))
    # projVect[0] * camera.v[2] - projVect[2] * camera.v[0] is the dot product between projvect's horizontal components
    # and a vector perpendicular to the camera vector
    x = int(HALF_WIDTH * (1 + (projVect[0] * camera.v[2] - projVect[2] * camera.v[0]) / k))

    return np.array([x, y])


def draw_line(p1, p2, color=BLACK, thickness=3):
    pygame.draw.line(screen, color, p1.screen_pos, p2.screen_pos, thickness)


def put_text(text, size, color, place):
    txt = pygame.font.SysFont('minecraftiaregular', size).render(text, True, color)
    screen.blit(txt, place)


def ptNearEdge(point, node1, node2):
    # returns True if point is in ellipse around nodes and not closer than NODE_SIZE to the nodes
    ellipse_ctt = 2
    d1 = np.linalg.norm(point - node1.screen_pos)
    if d1 < node1.size:
        return False

    d2 = np.linalg.norm(point - node2.screen_pos)
    if d2 < node2.size:
        return False

    d = np.linalg.norm(node1.screen_pos - node2.screen_pos)
    if d1 + d2 - d < ellipse_ctt:
        return True
    else:
        return False


def mouseRealPos(mouse_pos, ref_pt, camera):
    projVect = ref_pt - camera.pos
    k = np.dot(camera.v, projVect)
    mousePosPlane = np.multiply([(mouse_pos[0] - HALF_WIDTH) * camera.v[2],
                                 HALF_HEIGHT - mouse_pos[1],
                                 -(mouse_pos[0] - HALF_WIDTH) * camera.v[0]],
                                [1 / HALF_WIDTH, 1 / HALF_HEIGHT, 1 / HALF_WIDTH])

    return camera.pos + (camera.v + mousePosPlane) * k


def randomWalk(walk, counter, g):
    walk[3] -= 1
    if walk[2] == 0:
        # if walk in node
        walk[3] -= 2
    if walk[3] < 0:
        walk[3] = counter
        if walk[2] == 1:
            # if walk in edge
            walk[0] = walk[1]
            neighbors = [i for i in range(g.order) if g.adjMtx[walk[0], i] == 1]
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
adjMtx = get_random_graph(20, 0.15)

camera = Camera(np.array([-8., 0., 0.]), 0.)

nodes = [Node(i, np.random.rand(3) * 5, color=BLACK) for i in range(len(adjMtx))]
# random colors
# nodes = [Node(i, np.random.rand(3) * 1, color=np.random.randint(0, 256, 3)) for i in range(NODE_NUM)]

g = Graph(nodes, adjMtx)

center = g.getCenter()

nodesByDist = [i for i in range(len(adjMtx))]

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

x = 0
beat = False

game_over = False
while not game_over:
    clock.tick(FPS)
    screen.fill(WHITE)
    keys = pygame.key.get_pressed()
    mouse_button = (0, 0, 0)
    mouse_pos = np.array(pygame.mouse.get_pos())
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
                    if node.screen_pos is not False and selectRect.collidepoint(node.screen_pos):
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
                    unselectedNodes.sort(key=lambda x: g.listPosByIndex(x), reverse=True)
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

                    g.order += len(copyAdjMtx)
                    g.nodeNum += len(copyAdjMtx)
                    g.distMtx = np.zeros((g.order, g.order))
                    g.vectMtx = np.zeros((g.order, g.order, 3))
            if event.key == pygame.K_x:
                mode = (mode+1) % 2
            if event.key == pygame.K_SPACE:
                if mode == 0:
                    camera.rotate = not camera.rotate
            if event.key == pygame.K_b:
                beat = not beat

    if beat:
        x = (x+1/50) % 1
        ELASTIC_COEF = 0.03 + 0.02*np.cos(2*np.pi*x)

    #walk = randomWalk(walk, counter, g)

    #  UPDATE POSITION AND DISTANCE MATRICES
    g.updateMtxs()

    # UPDATE SIZES, SPEEDS AND POSITIONS OF NODES
    if draggedNode != NONE:
        index = g.listPosByIndex(draggedNode)
        mousePos3d = mouseRealPos(mouse_pos, g.nodes[index].pos, camera)
        pygame.draw.circle(screen, RED, project(mousePos3d), g.nodes[index].size - 1, 2)
        g.nodes[index].speed += (mousePos3d - g.nodes[index].pos) * 0.1 * FRICTION_COEF

    # GRAVITY
    '''
    for i in range(12):
        for j in range(12):
            pos = project(np.array([i/2, 0, j/2]))
            if pos is not False:
                pygame.draw.circle(screen, BLACK, pos, 1)

    for node in g.nodes:
        node.speed += np.array([0., -0.02, 0.])
        node.updateSize(camera.pos)
        node.updateSpeed(g.nodes)
        node.pos += node.speed
        if node.pos[1] < 0:
            node.pos[1] = 0
        node.screen_pos = project(node.pos)'''

    for node in g.nodes:
        node.updateSize(camera.pos)
        node.updateSpeed(g)
        node.pos += node.speed
        node.screen_pos = project(node.pos)

    # DRAW AND EDIT EDGES
    for i in range(g.order):
        for j in range(i, g.order):
            # if nodes joined and in screen
            if g.adjMtx[i, j] and g.nodes[i].screen_pos is not False and g.nodes[j].screen_pos is not False:
                # if mouse near edge
                if ptNearEdge(mouse_pos, g.nodes[i], g.nodes[j]):
                    # draw edge highlighted if mouse near it
                    draw_line(g.nodes[i], g.nodes[j], RED, 5)
                    if mouse_button[2]:  # right click
                        # delete edge
                        g.adjMtx[i, j] = 0
                        g.adjMtx[j, i] = 0
                    elif click[0] and not edgeEdited:  # if left click and no edge edited yet
                        # subdivide edge
                        g.subdivideEdge(i, j)
                        edgeEdited = True
                # if mouse not near edge
                else:
                    # draw edge normally
                    draw_line(g.nodes[i], g.nodes[j], BLACK)
                    '''
                    if i == walk[0] and j == walk[1] and walk[2] == 1:
                        draw_line(g.nodes[i], g.nodes[j], BLUE, 8)'''

    #  DRAW NODES
    # nodesByDist.sort(reverse=True, key=lambda n: g.nodes[n].cameraDist)
    # for i in nodesByDist:
    for node in g.nodes:
        # if node in screen
        if node.screen_pos is not False:
            node.draw()
            '''
            if g.listPosByIndex(node.idx) == walk[0] and walk[2] == 0:
                pygame.draw.circle(screen, BLUE, node.screen_pos, node.size + 1)'''
            mouseDist = np.linalg.norm(mouse_pos - node.screen_pos)
            # if mouse over node
            if mouseDist < node.size + 1:
                # draw highlight
                nodeHighlighted = True
                if click[0]:
                    draggedNode = node.idx
                pygame.draw.circle(screen, RED, node.screen_pos, node.size + 1, 2)
                if click[1]:  # right click
                    # if no selected nodes, select it
                    if not selectedNodes:
                        node.selected = True
                        selectedNodes.append(node.idx)
                    else:
                        if keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]:
                            # if there are selected nodes and control pressed, add or remove from selected nodes
                            node.selected = not node.selected
                            if node.selected:
                                selectedNodes.append(node.idx)
                            else:
                                selectedNodes.remove(node.idx)
                        else:
                            # if control not pressed, connect selected nodes to node
                            listPos = g.listPosByIndex(node.idx)
                            node.selected = False
                            for index in selectedNodes:
                                listPos2 = g.listPosByIndex(index)
                                g.connect(listPos, listPos2)
                                g.nodes[listPos2].selected = False
                            selectedNodes = []

    # MOUSE RECTANGLE SELECTION
    if click[0]:
        selectRectPt = mouse_pos
        selectRect = pygame.Rect(mouse_pos, [0, 0])
        if nodeHighlighted:
            drag = False
    if drag:
        # determine up-left and down-right pts of selection rectangle
        up_left = np.array([min(selectRectPt[0], mouse_pos[0]),
                            min(selectRectPt[1], mouse_pos[1])])
        down_right = np.array([max(selectRectPt[0], mouse_pos[0]),
                               max(selectRectPt[1], mouse_pos[1])])

        # get selection rectangle and draw it
        selectRect = pygame.Rect(up_left, [down_right[0]-up_left[0], down_right[1]-up_left[1]])
        pygame.draw.rect(screen, BLACK, selectRect, width=1)
        # highlight nodes in selection rectangle
        for node in g.nodes:
            # if node is in rectangle
            if node.screen_pos is not False and selectRect.collidepoint(node.screen_pos):
                pygame.draw.circle(screen, RED, node.screen_pos, node.size + 1, 2)

    # if right click no node or edge, unselect all nodes
    if click[1] and not nodeHighlighted and not edgeEdited:
        for node in selectedNodes:
            g.nodes[g.listPosByIndex(node)].selected = False
        selectedNodes = []

    if selectedNodes:
        # ADD NODE AND JOIN IT TO SELECTED ONES
        if click[0] and not (nodeHighlighted or edgeEdited or keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]):
            listPos = g.listPosByIndex(selectedNodes[0])
            g.nodes[listPos].selected = False
            mousePos3d = mouseRealPos(mouse_pos, g.nodes[listPos].pos, camera)
            g.addNode(mousePos3d)
            g.connect(listPos, g.order - 1)
            for node in selectedNodes[1:]:
                listPos = g.listPosByIndex(node)
                g.nodes[listPos].selected = False
                g.connect(listPos, g.order - 1)
            selected_nodes = []
        # DELETE NODE
        if keys[pygame.K_BACKSPACE]:
            for node in selectedNodes:
                g.deleteNode(node)
            selectedNodes = []
        # FIX NODE
        if keys[pygame.K_f]:
            for node in selectedNodes:
                listPos = g.listPosByIndex(node)
                g.nodes[listPos].selected = False
                g.nodes[listPos].fixed = not g.nodes[listPos].fixed
            selectedNodes = []

    if mode == 1:
        camera.move_free(keys)
        put_text("MODE: FREE", 16, BLACK, (8, 24))
    elif mode == 0:
        center = g.getCenter()
        camera.move_around(keys, center)
        put_text("MODE: POLAR", 16, BLACK, (8, 24))

    put_text("FPS: " + str(round(clock.get_fps(), 5)), 16, BLACK, (8, 8))
    # put_text(str(camera.pos), 16, BLACK, (8, 20))

    if show_indexes:
        for node in g.nodes:
            if node.screen_pos is not False:
                put_text(str(node.idx), 12, WHITE, node.screen_pos-np.array([4,5]))

    pygame.display.update()

pygame.quit()
quit()