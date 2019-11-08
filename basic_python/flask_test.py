from flask import Flask, escape, url_for, request, render_template,session, redirect
import flask as f
import os
# 设置单一模块或包名称
app = Flask(__name__)
# 设置会话秘钥
print(os.urandom(16))
app.secret_key = os.urandom(16)


@app.route('/logout')
def logout():
    session.pop('username')
    return f.redirect(f.url_for('page'))

@app.route('/home/')
@app.route('/home/<name>')
def page(name=None):
    return render_template('index.html', name=name)

@app.route('/')
def page_login():
    return f.render_template('login.html')

@app.route('/user/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username']=request.form['username']
        return f.redirect(f.url_for('page'))

# 传参数
@app.route('/user/<username>')
def show_user_name(username):
    return 'username:{}'.format(escape(username))

@app.route('/user2/<int:user_id>')
def show_user_id(user_id):
    return 'user_id:{}'.format(user_id)


@app.route('/user3/<path:sub_path>')
def show_sub_path(sub_path):
    return 'sub_path:{}'.format(escape(sub_path))



# with app.test_request_context():
#     print(url_for(show_user_name, username='/user/xiaonan'))
#     print(url_for(show_user_id, user_id='12'))
#     print(url_for(show_sub_path, sub_path='user/22'))

