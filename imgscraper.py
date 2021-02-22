
from bs4 import BeautifulSoup
import requests
import shutil
#global variables
urlArr = []
count = 1
url1 = 'https://westernmustangs.ca/sports/womens-basketball/roster'
url2 = 'https://westernmustangs.ca/sports/womens-soccer/roster'
url3 = 'https://westernmustangs.ca/sports/football/roster'
ogArr = [url1, url2, url3]




def scraper(url):
        response = requests.get(url, timeout=5)
        content = BeautifulSoup(response.content, "html.parser")

        for list in content.findAll('div', attrs={"class": "sidearm-roster-player-image column"}):
                #print(list)
                playerUrl = "https://westernmustangs.ca" + str(list)[str(list).find('href="')+6:str(list).find('" title')]
                response2 = requests.get(playerUrl, timeout=5)
                content2 = BeautifulSoup(response2.content, "html.parser")
                for imgdiv in content2.findAll('div', attrs={"class": "sidearm-roster-player-image"}):
                        #print(imgdiv)
                        imgurl = "https://westernmustangs.ca" + str(imgdiv)[str(imgdiv).find('src="')+5:str(imgdiv).find('"/>')]
                        #print(imgurl)
                        urlArr.append(imgurl)
                #print(playerUrl)

for urls in ogArr:
        scraper(urls)



#f = open("srcurl.txt", "w")
for url in urlArr:
        filename = "images/face" + str(count) + ".jpg"
        r = requests.get(url, stream = True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw, f)
                
            print('Image sucessfully Downloaded: ',filename)
        else:
            print('Image Couldn\'t be retreived')
            
        print(filename)
        count = count + 1
        #f.write(str(url))

        #f.write("\n")
#f.close()
