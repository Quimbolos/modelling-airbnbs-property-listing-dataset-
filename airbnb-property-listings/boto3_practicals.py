# %%

import boto3 

s3_client = boto3.client('s3')

response = s3_client.upload_file('IMG_0256.jpeg', 'myfirstbucketjbf', 'oli.jpg')
# %%

s3 = boto3.resource('s3')

my_bucket = s3.Bucket('myfirstbucketjbf')

for file in my_bucket.objects.all():
    print(file.key)

s3 = boto3.client('s3')

# Ofcourse, change the names of the files to match yours.
s3.download_file('myfirstbucketjbf', 'oli.jpg', 'oli.png')
# %%

import requests
# Change this with your URL
url = 'https://myfirstbucketjbf.s3.eu-west-2.amazonaws.com/oli.jpg'

response = requests.get(url)
with open('oli.png', 'wb') as f:
    f.write(response.content)


# %%
s3 = boto3.resource('s3')

my_bucket = s3.Bucket('myfirstbucketjbf')

urls = []

for file in my_bucket.objects.all():
    urls.append(file.key)

s3 = boto3.client('s3')

for i in range(len(urls)):
    urls[i] = 'https://myfirstbucketjbf.s3.eu-west-2.amazonaws.com/'+urls[i]

print(urls)

for name in urls:
    response = requests.get(name)
    with open('test', 'wb') as f:
        f.write(response.content)
# %%
