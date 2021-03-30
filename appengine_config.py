#file to specify where App Engine should look for 3rd party libraries
from google.appengine.ext import vendor

# Add any libraries installed in the "lib" folder.
vendor.add('lib')