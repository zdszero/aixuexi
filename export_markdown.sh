FILE="./content/constitution_principle/CPU/controller/index.md"

gsed -E \
  -e 's/\{\{<\s*(svg|img)\s*"(.*)".*>\}\}/![](\/Users\/dingzifeng\/Develop\/cs-kaoyan-grocery\2)/g' \
  -e '/\{\.table.*\}/d' \
  -e 's/\{\{.*anchored.*\}\}(.*)\{\{.*\}\}/__\1__/g' \
  -e 's/\{\{[^}]*\}\}//g' \
  -e 's/[^!]\[(.*)\]\((.*)\)/[\1](www.csgraduates.com\/\2)/g' \
  "$FILE"

