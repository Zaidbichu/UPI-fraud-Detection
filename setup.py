from setuptools import find_packages,setup
from typing import List
def get_requirements()->list[str]:
    requirements_list:list[str]=[]
    try:
        with open('requirements.txt','r') as file:
            read_lines=file.readline()
            ##reads the line one by one
            for read_line in read_lines:
                requirement=read_line.strip()
            #it is used to remove the spaces 
            if requirement and requirement!='-e .':
                requirements_list.append(requirement)
    except FileNotFoundError:
        print("the file does not exist")
    return requirements_list
setup(
    name="UPI_FRAUD",
    version="0.0.1",
    author='zaid',
    author_email='zaidbichu4@gmail.com',
    packages=find_packages(),
    install_requirements=get_requirements()

)

