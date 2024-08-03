import re
import scrapy

class MathgenealogySpider(scrapy.Spider):
    name = 'MathGenealogy'
    allowed_domains = ['www.mathgenealogy.org']
    start_urls = ['https://www.mathgenealogy.org/']

    def __init__(self, person=None, *args, **kwargs):
        super(MathgenealogySpider, self).__init__(*args, **kwargs)
        self.person = person

    def parse(self, response):
        return scrapy.FormRequest.from_response(
            response,
            formdata={"searchTerms": self.person},
            callback=self.parse_main
        )

    def get_id(self, url):
        return re.sub(".*\?id=", "", url)

    def parse_main(self, response):
        yield from self.parse_ancestor(response)
        yield from self.parse_descendant(response)

    def parse_ancestor(self, response):
        yield from self.parse_person(response)
        for href in response.xpath("//p[starts-with(.,'Advisor')]//a/@href").getall():
            yield {
                "from": self.get_id(href),
                "to": self.get_id(response.url)
            }
            yield response.follow(href, callback=self.parse_ancestor)

    def parse_descendant(self, response):
        yield from self.parse_person(response)
        for href in response.xpath("//td/a/@href").getall():
            yield {
                "from": self.get_id(response.url),
                "to": self.get_id(href)
            }
            yield response.follow(href, callback=self.parse_descendant)

    def parse_person(self, response):
        yield {
            "id": self.get_id(response.url),
            "person": response.xpath("normalize-space(//h2)").get(),
            "university": response.css("span[style*='color:']::text").get(),
            "year": response.xpath("normalize-space(//span[@style='margin-right: 0.5em']/text()[2])").get(),
            "thesisTitle": response.css("#thesisTitle::text").get().replace("\n","")
        }
