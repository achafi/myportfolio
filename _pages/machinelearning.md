---
layout : archive
permalink : /machine-learning/
title : "My machine learning projects"
author_profile : true
header :
   image : "./assets/images/view.jpeg"

---

{% include base_path %}
{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive.html %}
  {% endfor %}
{% endfor %}