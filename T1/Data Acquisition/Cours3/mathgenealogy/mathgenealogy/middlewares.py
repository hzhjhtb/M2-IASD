import os
import logging
import scrapy

from urllib.parse import urlparse

class MathgenealogyDownloaderMiddleware:
    def url2filename(self, url):
        path = url.replace("https://","")
        if path=="" or path[-1] == "/":
            path = path + "ROOT"
        return "cache/" + path

    def process_request(self, request, spider):
        if request.method != "GET":
            return None

        if urlparse(request.url).scheme == "file":
           return None

        filename = self.url2filename(request.url)
        if os.path.isfile(filename):
            logging.info("Getting "+request.url+" from "+filename)
            with open(filename, "rb") as inf:
                response=inf.read()
            return scrapy.http.TextResponse(body=response, url=request.url,
                                            request=request)
        else:
            # Continue processing
            return None

    def process_response(self, request, response, spider):
        filename = self.url2filename(request.url)

        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            logging.info("Storing "+request.url+" into "+filename)
            with open(filename, "wb") as out:
                out.write(response.body)
        return response
