---
layout : collections
permalink : /machine-learning/
title : "My machine learning projects"
author_profile : true
header :
   image : "./assets/images/view.jpeg"

---
<!-- start index.html body -->



{% for post in posts %}
    {% include archive-single.html %}
{% endfor %}
