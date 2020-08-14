---
layout : archive
permalink : /machine-learning/
title : "My machine learning projects"
author_profile : true
header :
   image : "./assets/images/view.jpeg"

---
<ul>
  {% for post in site.posts %}
    <li>
      <a href="https://achafi.github.io/myportfolio">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>