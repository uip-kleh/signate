import os

class Use_api:
    home = os.environ['HOME']
    data = os.path.join(home, 'data')
    signate = os.path.join(data, 'signate')

    def __init__(self):
        if not os.path.exists(self.data): os.mkdir(self.data)
        if not os.path.exists(self.signate): os.mkdir(self.signate)

    def disp_list(self):
        os.system('signate list')

    def download(self, competitionID):
        competition = os.path.join(self.signate, str(competitionID))
        os.system('signate download --competition-id=' + str(competitionID))
        if not os.path.exists(competition): os.mkdir(competition)
        os.system('mv *.csv ' + competition)
        os.system('mv *.zip ' + competition)
        os.system('mv *tsv ' + competition)

if __name__ == '__main__':
    use_api = Use_api()
    use_api.disp_list()
    competitionID = int(input('Input a competitionID which you wanna download:'))
    use_api.download(competitionID)
