all: output.svg

output.json:
	scrapy crawl MathGenealogy -a person="Pierre Senellart" -O $@

output.dot: output.json json2dot.py
	./json2dot.py $^ > $@

output.svg: output.dot
	dot -Tsvg $^ > $@

clean:
	$(RM) output.*

.PHONY: all clean
