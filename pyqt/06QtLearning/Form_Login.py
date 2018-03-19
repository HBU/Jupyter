from PyQt5.QtWidgets import QDialog, QMessageBox
from qtpy import QtWidgets
from Form_Main import MainDialog
from login import Ui_Dialog
import pymssql

class LoginDialog(QDialog,Ui_Dialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
    def btnOK(self):# 函数名在Qt中定义好
        login_user = self.lineEdit.text()
        login_password = self.lineEdit_2.text()
        result = self.DatabaseQuery(login_user,login_password)
        print('result ====== '+str(result))
        if (result == 1):
            self.FormMain = MainDialog() # 实例化主窗体
            self.FormMain.show() # 打开新窗体
            self.close()# 关闭当前窗口
        else:
            QMessageBox.warning(self,"警告","用户名或密码错误！",QMessageBox.Yes)
            self.lineEdit.setFocus()


    def btnCancel(self):# 函数名在Qt中定义好
        self.close()
    def DatabaseQuery(self,uid,pwd): # 在数据库中查询用户名密码，存在返回1，不存在返回0
        print("start DatabaseQuery:")
        conn = pymssql.connect(host='.', user='sa', password='sql', database='Test', charset="GBK")
        print("Database connected!")
        sqlstr = 'select count(*) from usertable where userid = \''+ str(uid) +'\' and password = \''+ str(pwd) +'\''
        try:
            cur = conn.cursor()
            cur.execute(sqlstr)
            count = cur.fetchall()
        except:
            print("Error: unable to fetch data")
        finally:
            print("End of SQL")
        cur.close()
        conn.close()
        print('count:'+ str(count[0][0]))
        return count[0][0]

if __name__=="__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FormLogin = LoginDialog()
    FormLogin.show()
    sys.exit(app.exec_())