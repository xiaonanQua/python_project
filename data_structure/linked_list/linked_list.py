'''
 链表：是由许多拥有相同类型的数据项按特定顺序排列而成的顺序表
 特性：在计算机内存中存储位置不连续且随机存放
 优点：在插入和删除时很方便，当插入时申请一块内存单元，当被删除时，会还给系统
 缺点：查询无法像静态数据那样，随机查找数据，查询时只能按序查找
'''

# 单链表（single linked list）

# 定义结点类
# 添加数据域和指针域属性
class Node(object):
    def __init__(self):
        self.data = 0 # 定义int型的数据域
        self.next = None # 定义指针域指向空

# 动态创建带头结点的单链表
# 步骤：1.动态分配新结点p的内存空间
#      2.将原链表尾部的指针（next）指向新元素所在的内存位置
#      3.将新元素的next设置成None
#      4.将当前节点(链表中的尾结点)指向新结点，更新当前结点（将新结点赋给当前结点）
def create_single_linked_list():
    head = Node() # 建立链表头结点
    head.next = None # 头结点初始无下一个结点
    choose = 0 # 用于用户选择判断

    while choose != 2:

        print('(1)增加，(2)退出')
        try:
            choose = int(input("请选择一个操作：")) # 强制转化成静态数据类型，int类型
        except ValueError:
            print("输入有误")
            print("请重新输入\n")
        
        if choose == 1:
            new_node = Node() # 实例一个新结点对象，即为新结点申请内存空间
            new_node.data = input("请输入一个值：") # 为数据域赋值
            new_node.next = None # 下一个结点指向空，亦使此结点为尾结点

            # 判断头结点有没有链接结点
            if head.next == None:
                cur = new_node  # 将新结点赋给当前结点（尾结点）
                head.next = cur  # 将头结点和当前结点（尾结点）链接起来
            else:
                cur.next = new_node  # 将新结点赋给当前结点（此时链表的尾结点）
                cur = new_node  # 更新当前结点（尾结点）
    
    return head  # 返回头结点，此时已经链接很多结点。


# 遍历单链表
def traverse_single_linked_list(linked_list):
    ptr = linked_list.next # 指向第一个结点（非头结点）
    # 判断当前结点是否为空
    while ptr != None:
        print(ptr.data) # 输出结点中数据域的值
        ptr = ptr.next # 指向下一个结点

'''
链表实例应用
输入学生数据，并建立单向链表。实现学生表的创建，插入，删除，查询，修改
'''


# 学生类结点
class Student(object):
    def __init__(self):
        self.num = ''  # 学号
        self.name = ''  # 姓名
        self.math = 0  # 数学
        self.english = 0  # 英语
        self.next = None  # 指向下一个结点


# 创建学生链表（带头结点）
def create_student():
    # 创建学生链表的头结点
    head = Student() 
    head.next = None
    cur = head
    choose = 0

    while choose != 2:
        print('1:增加，2：退出')
        try:
            choose = int(input('请输入你的选择'))
        except ValueError:
            print('输入错误')
            print('请重新输入')

        if choose == 1:
            new_data = Student()
            new_data.num = int(input('请输入学号：'))
            new_data.name = input('请输入姓名：')
            new_data.math = eval(input('输入数学成绩：'))
            new_data.english = eval(input('输入英语成绩：'))
            new_data.next = None

            # 判断头结点是否有链接点
            if head.next is None:
                cur = new_data
                head.next = cur
            else:
                cur.next = new_data
                cur = new_data

    return head


# 读取链表
def read_link_list(link_list):
    student = link_list.next
    while student is not None:
        print('学号：{}，姓名：{}，数学：{}，英语：{}'.
              format(student.num, student.name, student.math, student.english))
        student = student.next


# 添加结点
def insert_node(link_list, name):
    print('插入新结点')
    # 添加新结点
    new_node = Student()
    new_node.num = input('请输入学号')
    new_node.name = input('请输入姓名')
    new_node.english = input('请输入英语成绩：')
    new_node.math = input('请输入数学成绩：')
    new_node.next = None
    # 学生数据
    student = link_list

    choose = 0
    print('1：首节点，2：尾节点，3：特定插入,4:退出')
    while choose != 4:
        try:
            choose = int(input('请输入你的操作：'))
        except ValueError:
            print('输入错误')

        # 在首节点前插入
        if choose is 1:
            new_node.next = student.next
            student.next = new_node
        elif choose is 2:  # 在尾节点插入
            # 遍历到尾节点
            while student.next is not None:
                student = student.next
            student.next = new_node
        elif choose is 3:  # 新结点插入在name之后
            while student.next is not None:
                if student.name is name:
                    new_node.next = student.next
                    student.next = new_node
        read_link_list(student)

    return student


# 删除结点
def delete_node(link_list, num):
    head = link_list
    while head.next is not None:
        if head.next.num == num:
            head.next = head.next.next
        head = head.next



# 调用函数
# linked_list = create_single_linked_list()  # 创建带头结点的单链表
# traverse_single_linked_list(linked_list)  # 遍历单链表
student_link = create_student()
insert_node(student_link, 'xiaonan')