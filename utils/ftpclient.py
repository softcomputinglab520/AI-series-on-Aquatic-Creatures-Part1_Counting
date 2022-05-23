import ftplib
# import os

# host = '140.121.103.9'
class FTPClient():

    def __init__(self):
        host = '140.121.136.63'
        username = 'SCLAB520'
        password = 'NTOUAIsclab520'
        self.f = ftplib.FTP()
        self.f.connect(host, 21)
        self.f.login(username, password)

    def ftp_upload(self, filename, local, remote_path):
        file_remote = remote_path + filename
        file_local = local + filename
        bufsize = 1024
        fp = open(file_local, 'rb')
        self.f.storbinary('STOR '+file_remote, fp, bufsize)
        fp.close()

    def makedir(self, remote_path):
        try:
            self.f.mkd(remote_path)
        except ftplib.error_perm as e:
            e_str = str(e)
            print(e_str)

    def makedirs(self, remote_path):
        temp = remote_path.split('/')
        for i in range(len(temp)):
            if i == 0:
                self.makedir(temp[i])
            else:
                if len(temp[i]) > 0:
                    self.makedir('/'.join(temp[:i + 1]))

    def ftp_download(self, videoURL):
        file_name = videoURL.split('/')[-1]
        file_remote = videoURL
        file_local = './ftpdownload/' + file_name
        bufsize = 1024
        fp = open(file_local, 'wb')
        self.f.retrbinary('RETR %s' % file_remote, fp.write, bufsize)
        fp.close()
        return file_name
