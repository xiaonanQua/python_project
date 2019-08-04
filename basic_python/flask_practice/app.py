from flask import Flask, render_template, redirect, url_for, request

# 创建应用对象
app = Flask(__name__)

# 使用一个装饰器去连接url的函数
@app.route('/')
def home():
    return 'hello world'


@app.route('/welcome')
def welcome():
    return render_template('welcome.html')  # 提供一个静态文件模板


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.form['username'] is not 'xioanan' and request.form['password'] is not '123':
        error = 'please check out your username or password'
    else:
        return redirect(url_for(home))
    return render_template('login.html', error=error)


if __name__ == '__main__':
    # 使用run()函数启动服务
    app.run(debug=True)

