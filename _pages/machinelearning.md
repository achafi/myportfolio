---
layout : collections
permalink : /machine-learning/
title : "My machine learning projects"
author_profile : true
header :
   image : "./assets/images/view.jpeg"

---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.permalink }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>