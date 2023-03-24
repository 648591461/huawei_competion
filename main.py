import sys
# Robot_gui.exe -m maps/1.txt -c ./ "python main.py"
import numpy as np


def read_util_ok(initial=False):  # 该函数用于读取输入直到得到一个OK
    state = []
    tmp = input()
    while tmp != "OK":
        if not initial:
            tmp = tmp.split(" ")
        state.append(tmp)
        tmp = input()
    return state


def end_is_valid(start_tpye, end, id, agent):  # 检查目的地合法性
    for i in range(4):
        if i != id:
            if agent[i].task[2] == end and start_tpye == agent[i].task[0]:
                return True
    return False


def distance_computed(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Agent():
    def __init__(self, id):
        self.id = id
        self.pos_workbench_id = -1  # 所处工作台ID
        self.x = 0.
        self.y = 0.
        self.v = 0.
        self.orient = 0.
        self.task = [-1, -1, -1]  # 计划运输产品类型， 起点工作台, 终点工作台
        self.action_dict = {  # 字典转换表
        0: 'forward',
        1: 'rotate',
        2: 'buy',
        3: 'sell',
        4: 'destroy'
    }
        self.distance_pid = PIDController(10, 0.1, 0.01, 0, dt=1, MAX=6., MIN=-2.)
        self.angle_pid = PIDController(2, 0.1, 0.1, 0, dt=1, MAX=np.pi, MIN=-np.pi)
        self.start_set = {i: [] for i in range(1, 10)}  # 可选起点  TODO 起点冲突
        self.end_set = {i: [] for i in range(1, 10)}  # 可选终点  TODO 终点冲突
        self.require_set = {i: [] for i in range(1, 10)}  # 工作台的工作需求状态登记表
        self.range = []  # 智能体负责的工作台
        self.product_id = 0  # 携带的物品类型

    def adopt_action(self, actions):
        # 动作执行
        for i in range(len(actions)):
            if actions[i][0] == 0 or actions[i][0] == 1:
                print(self.action_dict[actions[i][0]], self.id, actions[i][1])
            else:
                print(self.action_dict[actions[i][0]], self.id)

    # 前往目的工作台  对于导航算法也有提升空空间
    def go_to_workbench(self, tar_id, tar_workbench, start = False):
        tar_x, tar_y = float(tar_workbench[0]), float(tar_workbench[1])
        # 计算偏航角
        yaw = self.clc_yaw(tar_x - self.x, tar_y - self.y)
        distance = distance_computed(self.x, tar_x, self.y, tar_y)
        # omega = self.angle_pid.update(yaw)
        # omega = sigmoid(omega)
        # angular_speed_range = 2 * np.pi
        # omega = angular_speed_range * omega - np.pi
        v = min(distance / 1.5, 1) * self.distance_pid.update(distance, 1 - abs(yaw) / np.pi)
        # self.adopt_action([[1, omega], [0, v]])
        # sys.stderr.write(str(self.id) + " " + str(yaw) + " " + str(omega) +  "   "  + str(v)  + "\n")

        # # 如果偏航角 > 3/200 * PI 减速 最大速转向
        if yaw > 3 * np.pi / 200:
            self.adopt_action([[1, np.pi], [0, 0.]])
        else:
            self.adopt_action([[1, yaw / 0.15], [0, v]])
        if int(self.pos_workbench_id) != int(tar_id) and self.v < 0.05:  # 防止堵塞
            self.adopt_action([[1, np.pi], [0, 666]])

        if int(self.pos_workbench_id) == int(tar_id):  # 到达目的地:
            # sys.stderr.write(str((self.pos_workbench_id)) + " " + str(tar_id) + "\n")
            self.distance_pid.set_pid()
            self.angle_pid.set_pid()
            self.adopt_action([[0, 0]])
            if start:
                if self.product_id == 0:
                    self.adopt_action([[2]])
                if self.task[0] > 3 or self.product_id != 0:
                    self.task[1] = -1
                    task_take_table[self.id] = [-1, -1]

            else:
                self.task[0] = -1
                self.task[2] = -1
                self.adopt_action([[3]])
                task_give_table[self.id] = [-1, -1]
        return not start and int(self.pos_workbench_id) == int(tar_id)

    # 计算偏航角
    def clc_yaw(self, tar_x, tar_y):
        a = np.array([np.cos(self.orient), np.sin(self.orient)])
        b = np.array([tar_x, tar_y])
        # 夹角cos值
        cos_ = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # 夹角sin值
        sin_ = np.cross(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        yaw = np.arctan2(sin_, cos_)
        return yaw

    # 状态更新
    def set_state(self, pos_workbench_id, orient, x, y, product_id, v_x, v_y):  # 设置xy的坐标
        self.pos_workbench_id = int(pos_workbench_id) + 1
        self.orient = float(orient)
        self.x = float(x)
        self.y = float(y)
        self.product_id = int(product_id)
        self.v = float(np.sqrt(float(v_x) ** 2 + float(v_y) ** 2))
        return

    # 分配任务
    def set_task(self, task_type, start, end):
        self.task = [task_type, start, end]
        return

    # 跟新辅助表
    def update_task_set(self, state):
        status_dict = {
            4: [1, 2],
            5: [1, 3],
            6: [2, 3],
            7: [4, 5, 6],
            8: [7],
            9: [i for i in range(1, 8)]
        }
        start_set = {i: [] for i in range(1, 10)}  # 可选起点  TODO 起点冲突
        end_set = {i: [] for i in range(1, 10)}  # 可选终点  TODO 终点冲突
        require_set = {i: [] for i in range(1, 10)}  # 工作台的工作需求状态登记表

        for i in self.range:
            wb_message = state[i]
            type_bench = int(wb_message[0])
            if wb_message[-1] == '1':  # 解决起点冲突
                if type_bench <= 3 or [type_bench, i] not in task_take_table:
                    start_set[type_bench].append(i)
            if type_bench > 3:
                tmp = [i]
                product_status = int(wb_message[-2])
                for j in status_dict[type_bench]:
                    if [j, i] not in task_give_table:  # 已经有产品在运输的路上了
                        if product_status & (1 << j) == 0:
                            end_set[j].append(i)
                            tmp.append(j)
                if len(tmp) > 1:
                    require_set[type_bench].append(tmp)
        for i in range(1, 10):
            if len(require_set[i]) > 1:
                require_set[i].sort(key=lambda x: [-len(x[1:]), x[0]])
        # sys.stderr.write("起点栏展示\n") for debug
        # for k, v in start_set.items():
        #     sys.stderr.write(str(k) + " ".join(str(v)) + "\n")
        self.start_set, self.end_set, self.require_set =  start_set, end_set, require_set

    # 更新管辖区域
    def update_range(self, range):
        self.range = range


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, dt, MAX, MIN):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.dt = dt
        self.error_sum = 0.
        self.prev_error = 0.
        self.MAX = MAX
        self.MIN = MIN
    def set_pid(self):
        self.error_sum = 0.
        self.prev_error = 0.
    def update(self, error, K=1):
        self.error_sum += error * self.dt
        d_error = (error - self.prev_error) / self.dt

        # PID output
        output = self.Kp * error + self.Ki * self.error_sum + self.Kd * d_error
        # if not ((output == self.MAX and error > 0) or (output == self.MIN and error < 0)):
        #     self.error_sum += error * self.dt
        self.prev_error = error
        return K * max(min(output, self.MAX), self.MIN)

task_give_table = [[-1, -1] for _ in range(4)]  # 送料表
task_take_table = [[-1, -1] for _ in range(4)]  # 取料表
TEST_WORKBRNCH = {
                1:
                    [[20, 21, 27, 28, 11, 12, 13, 16, 18, 22, 23, 24],
                    [4, 8, 9, 14, 11, 12, 13, 16, 18, 22, 23, 24],
                    [32, 33, 39, 40, 11, 12, 13, 16, 18, 22, 23, 24],
                    [32, 33, 39, 40, 4, 8, 9, 14, 11, 20, 21, 27, 28, 11, 12, 13]],
                2:
                    [[1, 2, 3, 4],
                    [1, 2, 3, 4, 13],
                    [23, 24, 25, 22],
                    [23, 24, 25, 22, 13]],
                3:
                    [[12, 13, 14, 15, 21, 23, 24],
                    [23, 24, 26, 30, 33, 34, 35],
                    [12, 13, 14, 15, 21, 23, 24, 17],
                    [23, 24, 26, 30, 33, 34, 35, 29]],
                4:
                    [[12, 14, 16, 18],
                     [11, 13, 15, 18],
                     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 18],
                     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 18]]
}

# 初始目的地
INITIAL_TASK = {
                1:
                    [[2, 42, 32],
                    [3, 43, 14],
                    [3, 43, 32],
                    [1, 1, 14]],
                2:
                    [[3, 9, 3],
                    [3, 8, 1],
                    [3, 17, 23],
                    [3, 18, 25]],
                3:
                    [[2, 17, 23],
                     [3, 29, 23],
                     [3, 29, 25],
                     [2, 17, 25]],
                4:
                    [[3, 8, 14],
                     [3, 10, 16],
                     [1, 3, 18],
                     [2, 6, 18]]
}

FLAG7_TABLE = [[0, 0, 0, 1],
               [0, 1, 0, 1],
               [0, 0, -1, -1],
               [0, 0, 1, 1]]  # 用于决定是否切换工作模式
if __name__ == '__main__':
    # 初始化工作台  每个工作台的位置都是独一无二的
    workbench = {i: [] for i in range(1, 10)}  # 工作台坐标
    cnt_workbench = [0 for i in range(10)]  # 同种类工作台数量记录
    # 初始化智能体
    agent = [Agent(id) for id in range(4)]
    # 获取地图
    map = read_util_ok(initial=True)
    bench_id = 1
    for row in range(100):
        for col in range(100):
            if map[row][col] != '.' and map[row][col] != 'A':
                workbench[int(map[row][col])].append([bench_id, 0.25 + 0.5 * col, 0.25 + 0.5 * (99 - row)])
                # sys.stderr.write(str(bench_id) + " " + str(0.25 + 0.5 * col) + ' ' + str(0.25 + 0.5 * (99 - row)) + '\n')
                bench_id += 1
    for k, v in workbench.items():
        cnt_workbench[k] = len(v)
    num_workbench = sum(cnt_workbench)  # 工作台总数
    if num_workbench == 43:  # 地图1
        test_workbench = TEST_WORKBRNCH[1]
        initial_task = INITIAL_TASK[1]
        END8 = 17
        FLAG7 = FLAG7_TABLE[0]
    elif num_workbench == 25:  # 地图2
        test_workbench = TEST_WORKBRNCH[2]
        initial_task = INITIAL_TASK[2]
        END8 = 11
        FLAG7 = FLAG7_TABLE[1]
    elif num_workbench == 50:  # 地图3
        test_workbench = TEST_WORKBRNCH[3]
        initial_task = INITIAL_TASK[3]
        END8 = 11
        FLAG7 = FLAG7_TABLE[2]
    else:
        test_workbench = TEST_WORKBRNCH[4]
        initial_task = INITIAL_TASK[4]
        END8 = 17
        FLAG7 = FLAG7_TABLE[3]
    # source_designation_table = {i: {} for i in range(num_workbench)}
    product_reservation_table = [i for i in range(1, num_workbench + 1)]  # 记录生产中的产品将运往那个工作台 主要针对4 5 6 [start, end] 订货制度
    for id in range(4):
        agent[id].update_range(test_workbench[id])  # 负责区域分配
        agent[id].set_task(initial_task[id][0], initial_task[id][1], initial_task[id][2])
        task_give_table[id] = [initial_task[id][0], initial_task[id][2]]
        task_take_table[id] = [initial_task[id][0], initial_task[id][1]]
    print('OK')  # 输出OK 表示初始化结束
    sys.stdout.flush()  # 别忘了flush一下标准输出
    try:
        while True:
            # 首先输出读入的第一个整数: 帧ID
            frame_id = input().split(' ')[0]
            print(frame_id)
            # sys.stderr.write(frame_id + "\n")
            # 忽略其他输入数据，读到OK为止
            state = read_util_ok()
            # for i in range(len(state)):
            #     sys.stderr.write(" ".join(state[i]) + "\n")
            for id in range(4):
                if sum(agent[id].task) != -3:
                    task_give_table[id] = [agent[id].task[0], agent[id].task[2]]
                if agent[id].task[1] != -1:
                    task_take_table[id] = [agent[id].task[0], agent[id].task[1]]
            for id in range(4):
                agent[id].set_state(state[id - 4][0], state[id - 4][7], state[id - 4][8], state[id - 4][9], state[id - 4][1], state[id - 4][5], state[id - 4][6])  # 更新 agent的坐标信息
                agent[id].update_task_set(state)  # 任务栏更新
            # 任务确定
            for id in range(4):
                # sys.stderr.write(" ".join(str(agent[id].task)) + " ")
                if sum(agent[id].task) == -3:  # 智能体空闲，进行任务分配
                    if FLAG7[id] == 1:  # 工作模式1: 需求优先
                        # 分配任务 前往指定地点
                        # 起点选择  倒序 优先选取贵重的物资 重点改进的地方
                        task_type, start, end = -1, -1, -1

                        if agent[id].start_set[7] != []:  # 7 -> 8/9
                            start = agent[id].start_set[7][-1]
                            end = END8
                            task_type = 7
                        else:
                        # 检查缺什么
                            if agent[id].require_set[7] == []:
                                for i in range(7, 0, -1):
                                    if agent[id].start_set[i] != [] and agent[id].end_set[i] != []:
                                        end = agent[id].end_set[i][-1]
                                        start = agent[id].start_set[i][-1]
                                        task_type = i
                                        break
                            else:
                                require_table = agent[id].require_set[7][-1]  # [id, require1, require2, require]
                                while len(require_table) <= 1:
                                    agent[id].require_set.pop()
                                    require_table = agent[id].require_set[7][-1]
                                # sys.stderr.write(" ".join(str(require_table)) + "\n")
                            #
                            #     # 检查有没有
                                if sum([len(agent[id].start_set[t]) for t in require_table[1:]]) > 0:  # 有的话 4 5 6 -> 7
                                    for i in range(len(require_table) - 1, 0, -1):  # TODO 可以优化一下 选择距离最近的哪一个
                                        if agent[id].start_set[require_table[i]] != []:  # TODO 可以优化一下 选择距离最近的哪一个
                                            start = agent[id].start_set[require_table[i]][-1]
                                            end = require_table[0]
                                            task_type = require_table[i]
                                            break

                                else:  # 没有的话 下一层 1 2 3 -> 4 5 6
                                    # sys.stderr.write(str(id) + ":" + " ".join(str(agent[id].require_set[require_table[1]])) + "\n")
                                    if agent[id].require_set[require_table[1]] == []:
                                        for i in range(7, 0, -1):
                                            if agent[id].start_set[i] != [] and agent[id].end_set[i] != []:
                                                end = agent[id].end_set[i][-1]
                                                start = agent[id].start_set[i][-1]
                                                task_type = i
                                                break
                                    else:
                                        require_table_456 = agent[id].require_set[require_table[1]][-1]  # 确定完成度最高的材料 [id, require1, require1]
                                        for require_type in reversed(require_table[1:]):  # 在缺的材料中遍历
                                            if agent[id].require_set[require_type] != []:
                                                rt = agent[id].require_set[require_type][-1]
                                                if len(rt) > 0 and len(rt) < len(require_table_456):
                                                    require_table_456 = rt
                                        # if int(frame_id) > 49:
                                        #     sys.stderr.write(str(id) + " ".join(str(require_table_456)) + "\n")
                                        distance_A = 250
                                        # sys.stderr.write(" ".join(str(require_table_456[-1])) + "\n")
                                        # sys.stderr.write("Agent_id:" + str(id) + " ".join(str(workbench[require_table_456[-1]])) + "\n")
                                        for i in range(len(workbench[require_table_456[-1]])):  # 需要划定范围
                                            s = workbench[require_table_456[-1]][i][0]
                                            tmp = distance_computed(float(state[s][1]), float(state[require_table_456[0]][1]),
                                                                    float(state[s][2]), float(state[require_table_456[0]][2]))
                                            if tmp < distance_A:
                                                distance_A = tmp
                                                start = s
                                        if start != -1:
                                            end = require_table_456[0]
                                            task_type = require_table_456[-1]
                    elif FLAG7[id] == 0:  # 工作模式2 专注生产
                        # 专注自身的工作台  目前只针对地图1的编写
                        for i in range(1, 8):
                            if agent[id].require_set[i] != []:
                                require_table = agent[id].require_set[i][-1]
                                break
                        distance_A = 250
                        # sys.stderr.write(" ".join(str(require_table_456[-1])) + "\n")
                        # sys.stderr.write("Agent_id:" + str(id) + " ".join(str(workbench[require_table_456[-1]])) + "\n")
                        for i in range(len(workbench[require_table[-1]])):  # 需要划定范围
                            s = workbench[require_table[-1]][i][0]
                            tmp = distance_computed(float(state[s][1]), float(state[require_table[0]][1]),
                                                    float(state[s][2]), float(state[require_table[0]][2]))
                            if tmp < distance_A:
                                distance_A = tmp
                                start = s
                        if start != -1:
                            end = require_table[0]
                            task_type = require_table[-1]
                    elif FLAG7[id] == -1:  # 工作模式3 专注送货
                        for i in range(7, 0, -1):
                            if agent[id].start_set[i] != []:
                                start = agent[id].start_set[i][-1]
                                task_type = i
                                end = 25  # map3
                                break

                    agent[id].set_task(task_type, start, end)
                if agent[id].task[1] != -1:
                    agent[id].go_to_workbench(agent[id].task[1], state[agent[id].task[1]][1:3], start=True)
                elif agent[id].task[2] != -1:
                    agent[id].go_to_workbench(agent[id].task[2], state[agent[id].task[2]][1:3])
            # sys.stderr.write("\n")
            #     sys.stderr.write("Agent_id:" + str(id) + " ".join(str(agent[0].task)) + "\n")

            print('OK')  # 当前帧控制指令结束，输出OK
            sys.stdout.flush()  # flush标准输出，以便判题器及时接收数据
    except EOFError:  # 读到EOF时, 程序结束
        pass

