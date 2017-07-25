import lxml.html as parser
from bs4 import BeautifulSoup 
import glob as glb

def convert_to_textsum_data_format(fname):
	h = open(fname, 'r')
	soup = BeautifulSoup(h.read(), "lxml")
	d = soup.find_all('div', {'class': 'meta'})
	content = ''
	for e in d: 
		t = e.find('div', {'class': 'list-title mathjax'})
		p = e.find('p', {'class': 'mathjax'})
		if t != None and p != None:
			#print(str(t.text).lstrip("\n").lstrip("Ttile:").rstrip("\n"))#,"***", p.text)#, e.next_sibling)
			t = " abstract=<d><p><s> " + t.text.lstrip("\n").lstrip("Title:") + " </s></p></d> "
			l = p.text.split('.')
			sent= ''
			for ll in l:
				sent = sent + " <s>" + ll + ". " + "</s> " 
			abs = " article=<d> <p> " + sent + "</p> </d> " 
			content = content + t + abs
			
	h.close()
	print(content)
	return content

file = "astroph_data.txt"
h = open(file, "w")
list = glb.glob('*.html')
print(list)
for f in list:
	print(f)
	cntnt = convert_to_textsum_data_format(f)
	h.write(cntnt)
h.close()
