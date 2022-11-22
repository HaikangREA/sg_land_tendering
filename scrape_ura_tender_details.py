# from fake_useragent import UserAgent
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
from random import choice
from datetime import datetime
import time
import random
import urllib3
import warnings
import bs4
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os
import shutil
import logging
import glob
import itertools
import re
from typing import List

# this line is to test the connection with github

# os.chdir(r'G:\\REA\\Working files\\land-bidding\\Table extraction')
with open('user_agent.txt') as f:
    ua_list = [ua.strip() for ua in f]
    f.close()

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='scraping.log', mode='w', encoding='utf-8')
formatter = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s")
handler.setFormatter(formatter)
root_logger.addHandler(handler)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")
original_wd = os.getcwd()


def random_time_delay():
    time.sleep(random.uniform(15, 30))


class WebScraper:

    def __init__(self, save_path: str = None, logger=None):
        self.save_path = save_path
        self.driver = self.open_browser()
        self.logger = logger

    def open_browser(self):
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument(f"user-agent={choice(ua_list)}")
        chrome_options.add_experimental_option('prefs',
                                               {"download.default_directory": self.save_path,
                                                "download.prompt_for_download": False,
                                                "download.directory_upgrade": True,
                                                "plugins.always_open_pdf_externally": True
                                                }
                                               )
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
        return driver

    def log(self, msg: str, level='info'):
        print(msg)
        if self.logger:
            try:
                if level == 'warning':
                    self.logger.warning(msg)
                elif level == 'error':
                    self.logger.error(msg)
                else:
                    self.logger.info(msg)
            except:
                pass
        return msg

    def search(self, land_parcel: str):
        self.log(f'{self.land_parcel}: Searching...')
        try:
            # go to home page of ura gov site
            self.driver.get('https://www.ura.gov.sg/maps/')
            random_time_delay()

            # click on "view government land sales site"
            self.driver.find_element(by=By.XPATH,
                                     value='//*[@id="us-c-ip"]/div[1]/div[1]/div[4]/div[3]/div[6]/div[2]').click()
            random_time_delay()

            # search land parcel
            self.driver.find_element(by=By.CLASS_NAME, value='us-s-txt').send_keys(land_parcel)

        except:
            self.log(f'{land_parcel}: Error occurred when searching', 'error')

    def extract_url(self):
        source = self.driver.page_source
        soup = bs4.BeautifulSoup(source, 'html.parser')
        url = None

        for i in soup.find_all('a'):
            try:
                if ('Tender-Results' in i['href']):
                    url = i['href']
                    random_time_delay()
                    break
            except:
                pass

        return url

    def get_url(self, land_parcel: str) -> [List[str], List[str], List[str]]:
        self.land_parcel = land_parcel

        try:
            self.search(land_parcel)
            random_time_delay()

            # get the number of searched results
            results = self.driver.find_elements(by=By.XPATH, value='//a[@data-parentid="0" and @data-type="service"]')

            url_list = []
            awardDates = []
            nameList = []
            if len(results) == 0:
                self.log(f"{self.land_parcel}: No URA sales site result", "warning")

            elif len(results) == 1:
                self.log(f"{self.land_parcel}: 1 URA sales site searched")
                # Click on result
                random_time_delay()
                self.driver.find_element(by=By.CLASS_NAME, value='us-sr-result').click()

                # Get url for Tender Result pdf
                self.log(f"{self.land_parcel}: Extracting URL...")
                random_time_delay()
                url = self.extract_url()
                if url and pd.notna(url):
                    url_list.append(url)
                    self.log("Done")
                else:
                    self.log("Invalid URL", "warning")
                random_time_delay()

                # get the exact land parcel name and the dates of award
                try:
                    land_parcel_name = self.driver.find_element(by=By.CLASS_NAME, value="us-ip-poi-a-title").text
                    random_time_delay()
                except:
                    land_parcel_name = self.land_parcel

                try:
                    award_date = self.driver.find_element(by=By.XPATH,
                                                          value='//*[@id="us-c-ip"]/div[3]/div[1]/div[2]/div[1]/div[2]/div[1]/div[2]/div[4]/div[2]').text
                    award_date = datetime.strptime(award_date, '%d %B %Y').strftime("%Y%m%d")
                except:
                    award_date = 'Unknown'

                awardDates.append(award_date)
                nameList.append(land_parcel_name)


            else:
                self.log(f"{self.land_parcel}: {len(results)} URA sales sites searched")
                for id_ in range(len(results)):  # 0 1, len = 2
                    self.log(f"{self.land_parcel}: Extracting #{id_ + 1} URL...")

                    # click one by one
                    random_time_delay()
                    self.driver.find_element(by=By.XPATH,
                                             value=f'//a[@data-type="service" and @data-id="{id_}"]').click()

                    # Get url for Tender Result pdf
                    random_time_delay()
                    url = self.extract_url()
                    if url and pd.notna(url):
                        url_list.append(url)
                        self.log("Done")
                    else:
                        self.log("Invalid URL", "warning")
                    random_time_delay()

                    # get the exact land parcel name and the dates of award
                    try:
                        land_parcel_name = self.driver.find_element(by=By.CLASS_NAME, value="us-ip-poi-a-title").text
                        random_time_delay()
                    except:
                        land_parcel_name = self.land_parcel

                    try:
                        award_date = self.driver.find_element(by=By.XPATH,
                                                              value='//*[@id="us-c-ip"]/div[3]/div[1]/div[2]/div[1]/div[2]/div[1]/div[2]/div[4]/div[2]').text
                        award_date = datetime.strptime(award_date, '%d %B %Y').strftime("%Y%m%d")
                    except:
                        award_date = 'Unknown'

                    awardDates.append(award_date)
                    nameList.append(land_parcel_name)

                    # get back to the search results
                    if id_ < len(results) - 1:  # 1
                        random_time_delay()
                        self.search(land_parcel)

            self.log(f"{self.land_parcel}: {len(url_list)} valid URLs retrieved")
            return [url_list, awardDates, nameList]

        except:
            self.log(f"{self.land_parcel}: Error occurred when getting URL list", "error")
            pass

    def download(self, url_list: List[str], id_list: List[str] = None, name_list: List[str] = None):
        if not os.path.exists(os.path.join(self.save_path, 'temp')):
            os.makedirs(os.path.join(self.save_path, 'temp'))
        destination = os.path.join(self.save_path, 'temp')
        if not id_list:
            id_list = list(range(len(url_list)))

        if not name_list:
            name_list = [self.land_parcel]*len(url_list)

        for i in range(len(url_list)):
            _id_ = id_list[i]
            _exactName_ = name_list[i]

            # make sure the save path has no pdf file (just serve as a mid-point), if not, move these files to another folder
            existing_pdf = [file for file in os.listdir(self.save_path) if '.pdf' in file]
            if len(existing_pdf) > 0:
                if not os.path.exists(os.path.join(self.save_path, 'redundant')):
                    os.makedirs(os.path.join(self.save_path, 'redundant'))
                redundant_path = os.path.join(self.save_path, 'redundant')

                for file in existing_pdf:
                    source_redundant = os.path.join(self.save_path, file)
                    desti_redundant = os.path.join(redundant_path, file)
                    shutil.move(source_redundant, desti_redundant)

            random_time_delay()
            try:
                self.driver.get(url_list[i])

                random_time_delay()
                pdf = [file for file in os.listdir(self.save_path) if '.pdf' in file]
                if len(pdf) > 0:
                    pdf_file = pdf[0]
                    if len(pdf) > 1:
                        self.log(f"{self.land_parcel}_{i}: Multiple PDF downloaded or redundant files, chose the first one", "warning")
                    source_path = os.path.join(self.save_path, pdf_file)
                    # remove illegal punc in filename
                    illegal_punc = '[/\:*?"<>|]'

                    try:
                        file_name = re.sub(illegal_punc, '+', _exactName_)
                    except:
                        file_name = re.sub(illegal_punc, '+', self.land_parcel)

                    full_file_name = f"{file_name}_{_id_}.pdf"

                    # make sure there's no duplicated file name
                    filelist = os.listdir(destination)
                    occurrence = filelist.count(full_file_name)
                    if occurrence:
                        k = 0
                        full_file_name = f"{file_name}_{_id_}_0.pdf"
                        while filelist.count(full_file_name):
                            k += 1
                            full_file_name = f"{file_name}_{_id_}_{k}.pdf"

                    desti_path = os.path.join(destination, full_file_name)
                    shutil.move(source_path, desti_path)
                    self.log(f"{self.land_parcel}_{i}: Tender details saved in <{full_file_name}>")

                else:
                    self.log(f"{self.land_parcel}_{i}: No PDF downloaded", "warning")
            except:
                self.log(f"{self.land_parcel}_{i}: Error occurred when downloading", "error")

        self.log(f'{self.land_parcel}: Process ended', '\n')

    def scrape(self, landParcels: List[str]):
        try:
            for land_parcel in tqdm(landParcels):
                [url_list, id_list, name_list] = self.get_url(land_parcel)
                self.download(url_list, id_list, name_list)
            self.log("All done")
            self.driver.quit()
        except:
            self.log(f'{self.land_parcel}: Error occurred when passing URL to download function', "error")


if __name__ == "__main__":
    # read in land parcel list
    gls = pd.read_csv(r'G:\REA\Working files\land-bidding\land_sales_full_data\ready for uploading\gls_no_detail.csv')
    ura_gls = gls[gls.source == 'ura'].reset_index(drop=True)
    landParcels = list(ura_gls.land_parcel.reset_index(drop=True).apply(lambda x: x.replace('/', ' / ')).unique())

    # start scraping
    save_path = r'G:\REA\Working files\land-bidding\Table extraction\tenderer_details_ura'
    scraper = WebScraper(save_path=save_path, logger=root_logger)
    scraper.scrape(landParcels)
